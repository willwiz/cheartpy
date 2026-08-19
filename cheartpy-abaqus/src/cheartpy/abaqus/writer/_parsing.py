import argparse
from pathlib import Path

from pydantic import BaseModel, Field, ValidationError
from pytools.result import Err, Ok, Result

parser = argparse.ArgumentParser()
parser.add_argument("--region-mask", "--mask", type=Path, help="Path to the region mask file")
parser.add_argument("--prefix", "-p", type=Path, help="Path to the output file")
parser.add_argument("mesh", type=Path)


class ArgumentParser(BaseModel):
    mesh: Path = Field(..., description="Path to the mesh file")
    region_mask: Path | None = Field(None, description="Path to the region mask file")
    prefix: Path | None = Field(None, description="Path to the output file")


def get_cmdline_args(args: list[str] | None = None) -> Result[ArgumentParser]:
    """Parse command line arguments and return an ArgumentParser instance."""
    try:
        parsed_args = ArgumentParser.model_validate(vars(parser.parse_args(args)))
    except ValidationError as e:
        return Err(e)
    return Ok(parsed_args)
