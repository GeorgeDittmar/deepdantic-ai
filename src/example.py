from pydantask.models import Plan, TaskItem, CapabilityDescription
from pydantask.agents import DeepAgent
from pydantask.capabilities.runner_v2 import as_runner
from dotenv import load_dotenv, find_dotenv
from pprint import pprint
from pathlib import Path
import asyncio

load_dotenv(find_dotenv())

checkpoint_dir = Path("_checkpoint") / "primative"

async def write_to_file(content:str, filename:str) -> str:

    with open(f"tmp/{filename}", "w") as f:
        f.write(content)

    return f"{filename} was written to disk at location tmp/{filename}"

async def read_from_file(filename:str) -> str:
    with open(f"tmp/{filename}", "r") as f:
        return f.read()

writing_capability = CapabilityDescription(name="write_to_file", 
                                           description="Tool to write content to a file on disk. Use when there is output needing to be saved for a subtask, or a final output.",
                                           tool_func=as_runner(write_to_file))

reading_capability = CapabilityDescription(name="read_from_file",
                                           description="Tool to read the contents of a text file on disk. Use this when something needs to be loaded or read for a task.",
                                           tool_func=as_runner(read_from_file))
da = DeepAgent(
    # "I need a report of the news for today. Give me a high level summary and then a detailed version of major pieces of news as it pertains to the US and world. Output the report as markdown withj citations in the report. You must cite all sources at the end of the article.",
    # """I am testing your Deep Agent ability to plan and execute on an objective. I want to test your ability to create plans.
    # Create 2 Research tasks. Besure to have mentioned in the instructions to not actually search but make something up.
    #   - 1: Pretend to Research about the man on the moon.
    #   - 2: Pretend to Research why stone henge is green.
    # Then have a producer task that requires task 1 and 2 to be completed. Have the producer write a little song about cats in space.
    # Then have a conditional check to see which research result from task 1 and 2 is longer. If the lenght of task 1 is larger than task 2, create either a task about hollow moon. Otherwise create a about quantom physics.
    # Finally take the song and whichever research topic passed the conditional check and combine them into a single output in song form. Jsut have the song do not let the producer agent output anything else with its task.
    # Again this is a test to see how well you follow creating a plan.""",
    # "Research for me `pydantask` harness. It is a python project. Give me a report on what it is, what problem does it try to solve, its features, and an example usage. As well include a bio summary on the main author of the harness. Write this up in markdown format and be sure to cite your sources.",
    "Write a short story about scifi space travel. This should be saved to a file. Next take the contents of that file and pass it as inputs to another file I want you to write but translated to German.",
    model="gpt-5.4",
    trace=True,
    max_steps=10,
    default_capabilities_enabled=True,
    # capabilities=[writing_capability, reading_capability],
    checkpoint=True,
    checkpoint_dir=checkpoint_dir,
)

result = asyncio.run(da.run())

print(f"Checkpoint events saved to: {da.checkpoint_path}")

# pprint(result.model_dump())
# Write JSON data to a file

with open("Compression_test_2.json", "w", encoding="utf-8") as json_file:
    json_file.write(result.model_dump_json(indent=2))

final_output = result.final_result.detailed_output if result.final_result else ""
with open("Compression_test_2.md", "w", encoding="utf-8") as f:
    f.write(final_output)

final_output = result.final_result.detailed_output if result.final_result else ""

pprint(final_output or "<no final result>")
