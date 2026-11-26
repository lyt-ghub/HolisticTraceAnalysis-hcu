# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import BinaryIO
from uuid import uuid4


@contextmanager
def atomic_binary_output(output: Path) -> Iterator[BinaryIO]:
    """Publish a complete file with permissions inherited from its destination."""
    # A private temporary directory can give the file a restrictive Windows ACL
    # that survives a same-volume rename. Create the temporary file here instead.
    temporary = output.with_name(f".hta-{uuid4().hex}.tmp")
    handle = temporary.open("xb")
    try:
        with handle:
            yield handle
        temporary.replace(output)
    finally:
        temporary.unlink(missing_ok=True)
