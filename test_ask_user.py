import chainlit as cl


@cl.on_chat_start
async def main():
    res = await cl.AskActionMessage(
        content="Pick an action!",
        actions=[
            cl.Action(name="continue", value="continue", label="✅ Continue"),
            cl.Action(name="cancel", value="cancel", label="⏰prof.sincak-db⏰"),
        ],
    ).send()

    if res and res.get("value") == "continue":
        await cl.Message(
            content="Continue!",
        ).send()