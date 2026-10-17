# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""MkDocs hooks for TorchTPU documentation."""

import re

_MKDOCS_COMMENT_RE = re.compile(r"<!--\s*mkdocs:(.*?)\s*-->", re.DOTALL)


def on_page_markdown(markdown: str, **kwargs) -> str:
  """Uncomments `<!--mkdocs:...-->` directives for MkDocs rendering."""
  del kwargs
  return _MKDOCS_COMMENT_RE.sub(r"\1", markdown)
