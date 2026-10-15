from typing import Union, List, Dict, Any, Optional
import asyncio
import json
from loguru import logger
import numpy as np

from .conversation_utils import (
    create_batch_input,
    process_agent_output,
    send_conversation_start_signals,
    process_user_input,
    finalize_conversation_turn,
    cleanup_conversation,
    EMOJI_LIST,
)
from .types import WebSocketSend
from .tts_manager import TTSTaskManager
from ..chat_history_manager import store_message
from ..service_context import ServiceContext
from ..interpreter.service import shared_service, speaks_pattani

# Import necessary types from agent outputs
from ..agent.output_types import SentenceOutput, AudioOutput, DisplayText, Actions

# Translator greeting (Thai + Jawi) — used in translation mode when the
# frontend sends the "ai-speak-signal" (proactive speak) trigger. The stock
# English prompt (prompts/utils/proactive_speak_prompt.txt) would otherwise
# be machine-translated to an odd Thai sentence ("ขอพูดอะไรที่น่าสนใจ...").
# Only the Thai line is spoken (the TTS voice is Thai); both lines are shown.
TRANSLATOR_GREETING_TEXT_THAI = "ต้องการให้ฉันช่วยแปลภาษาคำไหนแจ้งได้เลย"
TRANSLATOR_GREETING_TEXT_JAWI = "کلاو نق ساي تولوڠ ترجمه بهاس، کبو ساي"
TRANSLATOR_GREETING_TEXT = (
    TRANSLATOR_GREETING_TEXT_THAI + " \n" + TRANSLATOR_GREETING_TEXT_JAWI
)


async def process_single_conversation(
    context: ServiceContext,
    websocket_send: WebSocketSend,
    client_uid: str,
    user_input: Union[str, np.ndarray],
    images: Optional[List[Dict[str, Any]]] = None,
    session_emoji: str = np.random.choice(EMOJI_LIST),
    metadata: Optional[Dict[str, Any]] = None,
) -> str:
    """Process a single-user conversation turn

    Args:
        context: Service context containing all configurations and engines
        websocket_send: WebSocket send function
        client_uid: Client unique identifier
        user_input: Text or audio input from user
        images: Optional list of image data
        session_emoji: Emoji identifier for the conversation
        metadata: Optional metadata for special processing flags

    Returns:
        str: Complete response text
    """
    # Create TTSTaskManager for this conversation
    tts_manager = TTSTaskManager()
    full_response = ""  # Initialize full_response here

    try:
        # Send initial signals
        await send_conversation_start_signals(websocket_send)
        logger.info(f"New Conversation Chain {session_emoji} started!")

        # Process user input
        input_text = await process_user_input(
            user_input, context.asr_engine, websocket_send
        )

        quick_phrase = metadata.get("quick_phrase") if metadata else None
        if quick_phrase:
            # Picked from the quick bar, not typed in the page: show it as
            # what the user said.
            await websocket_send(
                json.dumps({"type": "user-input-transcription", "text": input_text})
            )

        # Create batch input
        batch_input = create_batch_input(
            input_text=input_text,
            images=images,
            from_name=context.character_config.human_name,
            metadata=metadata,
        )

        # Store user message (check if we should skip storing to history)
        skip_history = metadata and metadata.get("skip_history", False)
        if context.history_uid and not skip_history:
            store_message(
                conf_uid=context.character_config.conf_uid,
                history_uid=context.history_uid,
                role="human",
                content=input_text,
                name=context.character_config.human_name,
            )

        if skip_history:
            logger.debug("Skipping storing user input to history (proactive speak)")

        logger.info(f"User input: {input_text}")
        if images:
            logger.info(f"With {len(images)} images")

        try:
            # Check if we should use translation mode (bypass agent and translate user input)
            translating = (
                context.character_config.use_translation
                and context.translate_engine is not None
            )
            tts_engine = context.tts_engine
            if quick_phrase or translating:
                if quick_phrase:
                    # A standard sentence of the sentence bank: say it in
                    # the other language as the bank has it. Nothing is
                    # translated, and the agent is not asked.
                    tts_text = quick_phrase["spoken"]
                    output_text = quick_phrase["display"]
                    logger.info(
                        f"Quick phrase ({quick_phrase['direction']}): {tts_text}"
                    )
                    if not speaks_pattani(context):
                        # this character's voice cannot read Thai script
                        tts_engine = await asyncio.to_thread(shared_service().voice)
                elif metadata and metadata.get("proactive_speak"):
                    logger.info("Translation mode enabled: bypassing agent and translating user input")
                    # Proactive speak trigger: greet the user bilingually
                    # instead of translating the default English prompt.
                    # Only the Thai line is spoken by the TTS voice.
                    output_text = TRANSLATOR_GREETING_TEXT
                    tts_text = TRANSLATOR_GREETING_TEXT_THAI
                    logger.info(f"Translator greeting: {output_text}")
                else:
                    logger.info("Translation mode enabled: bypassing agent and translating user input")
                    # Translate the user's input text (in a thread: local
                    # models / LLM calls would block the event loop for seconds)
                    output_text = await asyncio.to_thread(
                        context.translate_engine.translate, input_text
                    )
                    tts_text = output_text
                    logger.info(f"Translated text: {output_text}")
                    # What is shown may carry more than what is spoken
                    # (e.g. the same sentence in Latin letters)
                    output_text = context.translate_engine.display_text(output_text)
                # Create display text for the translated text (to be shown in UI)
                display_text = DisplayText(
                    text=output_text,
                    name=context.character_config.character_name,
                    avatar=context.character_config.avatar,
                )
                # Create empty actions
                actions = Actions()
                # Create a SentenceOutput with the translated text
                sentence_output = SentenceOutput(
                    display_text=display_text,
                    tts_text=tts_text,
                    actions=actions,
                )
                # Process the sentence output (but disable translation in process_agent_output to avoid double translation)
                response_part = await process_agent_output(
                    output=sentence_output,
                    character_config=context.character_config,
                    live2d_model=context.live2d_model,
                    tts_engine=tts_engine,
                    websocket_send=websocket_send,
                    tts_manager=tts_manager,
                    translate_engine=None,  # Disable translation to avoid double translation
                )
                # Ensure response_part is treated as a string before concatenation
                response_part_str = (
                    str(response_part) if response_part is not None else ""
                )
                full_response += response_part_str  # Accumulate text response
            else:
                # Fallback to normal agent processing
                logger.info("Using normal agent processing")
                # agent.chat yields Union[SentenceOutput, Dict[str, Any]]
                agent_output_stream = context.agent_engine.chat(batch_input)

                async for output_item in agent_output_stream:
                    if (
                        isinstance(output_item, dict)
                        and output_item.get("type") == "tool_call_status"
                    ):
                        # Handle tool status event: send WebSocket message
                        output_item["name"] = context.character_config.character_name
                        logger.debug(f"Sending tool status update: {output_item}")

                        await websocket_send(json.dumps(output_item))

                    elif isinstance(output_item, (SentenceOutput, AudioOutput)):
                        # Handle SentenceOutput or AudioOutput
                        response_part = await process_agent_output(
                            output=output_item,
                            character_config=context.character_config,
                            live2d_model=context.live2d_model,
                            tts_engine=context.tts_engine,
                            websocket_send=websocket_send,  # Pass websocket_send for audio/tts messages
                            tts_manager=tts_manager,
                            translate_engine=context.translate_engine,
                        )
                        # Ensure response_part is treated as a string before concatenation
                        response_part_str = (
                            str(response_part) if response_part is not None else ""
                        )
                        full_response += response_part_str  # Accumulate text response
                    else:
                        logger.warning(
                            f"Received unexpected item type from agent chat stream: {type(output_item)}"
                        )
                        logger.debug(f"Unexpected item content: {output_item}")

        except Exception as e:
            logger.exception(
                f"Error processing agent response stream: {e}"
            )  # Log with stack trace
            await websocket_send(
                json.dumps(
                    {
                        "type": "error",
                        "message": f"Error processing agent response: {str(e)}",
                    }
                )
            )
            # full_response will contain partial response before error
        # --- End processing agent response ---

        # Wait for any pending TTS tasks
        if tts_manager.task_list:
            await asyncio.gather(*tts_manager.task_list)
            await websocket_send(json.dumps({"type": "backend-synth-complete"}))

        await finalize_conversation_turn(
            tts_manager=tts_manager,
            websocket_send=websocket_send,
            client_uid=client_uid,
        )

        if context.history_uid and full_response:  # Check full_response before storing
            store_message(
                conf_uid=context.character_config.conf_uid,
                history_uid=context.history_uid,
                role="ai",
                content=full_response,
                name=context.character_config.character_name,
                avatar=context.character_config.avatar,
            )
            logger.info(f"AI response: {full_response}")

        return full_response  # Return accumulated full_response

    except asyncio.CancelledError:
        logger.info(f"🤡👍 Conversation {session_emoji} cancelled because interrupted.")
        raise
    except Exception as e:
        logger.error(f"Error in conversation chain: {e}")
        await websocket_send(
            json.dumps({"type": "error", "message": f"Conversation error: {str(e)}"})
        )
        raise
    finally:
        cleanup_conversation(tts_manager, session_emoji)
