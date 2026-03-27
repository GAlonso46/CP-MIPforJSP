"""
Test that generates a small random instance, converts it to TA format, writes it to
instances/basic_jssp/ta_test, then parses it back and performs basic consistency checks.

Run: python3 tests/test_convert_and_parse.py
"""
from instance_generators.random_jssp import generate_instance
from utils.convert_to_ta import write_instance_to_ta_file, convert_instance_to_ta_format
from utils.parse_ta import parse_ta_file
import os


def main():
    inst = generate_instance(3, 4, seed=123)

    out_dir = "instances/basic_jssp"
    filename = "ta_test"
    written = write_instance_to_ta_file(inst, out_dir, filename)
    print(f"Wrote TA file: {written}")

    parsed = parse_ta_file(written)
    print("Parsed data keys:", sorted(parsed.keys()))

    # Basic checks
    assert parsed["machines"] is not None
    assert len(parsed["tasks"]) == 3 * 4
    # check precedence arcs count
    expected_arcs = 3 * (4 - 1)
    assert len(parsed["P"]) == expected_arcs

    print("Test completed successfully")


if __name__ == "__main__":
    main()
