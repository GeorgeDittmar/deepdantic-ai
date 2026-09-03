from pydantask.models import Plan, TaskItem, CapabilityDescription
from pydantask.agents import DeepAgent
from pydantask.capabilities.runner_v2 import as_runner
from dotenv import load_dotenv, find_dotenv
from pprint import pprint
from pathlib import Path
import asyncio

load_dotenv(find_dotenv())

checkpoint_dir = Path("_checkpoint") / "extraction_test"

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
    # Simple task to verify the extraction pipeline:
    # Create a haiku about autumn leaves, save it, then translate to French.
    "Calculate the sum of all prime numbers less than 100. Write the list of those primes to primes.txt. Then add up the squares of each of those primes and write that total sum to prime_squares_total.txt.",
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

with open("extraction_test.json", "w", encoding="utf-8") as json_file:
    json_file.write(result.model_dump_json(indent=2))

final_output = result.final_result.detailed_output if result.final_result else ""
with open("extraction_test.md", "w", encoding="utf-8") as f:
    f.write(final_output)

final_output = result.final_result.detailed_output if result.final_result else ""

pprint(final_output or "<no final result>")
