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
_FENCE_RE = re.compile(r"^([ ]*)(```+|~~~+)")
_CALLOUT_START_RE = re.compile(
    r"^([ ]*)>[ ]*(?:"
    r"\*\*(Note|Important|Warning|Tip|Caution):\*\*"
    r"|"
    r"\\?\[!(NOTE|IMPORTANT|WARNING|TIP|CAUTION)\\?\]"
    r")[ ]*(.*)$",
    re.IGNORECASE,
)
_BLOCKQUOTE_LINE_RE = re.compile(r"^[ ]*>[ ]?(.*)$")


def _convert_blockquote_callouts(markdown: str) -> str:
  """Converts `> **Note:** ...` and `> [!NOTE]` blockquotes to MkDocs admonitions."""
  lines = markdown.split("\n")
  out: list[str] = []
  in_fence = False
  fence_marker = ""
  i = 0

  while i < len(lines):
    line = lines[i]
    fence_match = _FENCE_RE.match(line)
    if fence_match:
      marker = fence_match.group(2)
      if not in_fence:
        in_fence = True
        fence_marker = marker
      elif (
          marker[0] == fence_marker[0]
          and len(marker) >= len(fence_marker)
          and not line[fence_match.end() :].strip()
      ):
        in_fence = False
        fence_marker = ""
      out.append(line)
      i += 1
      continue

    if not in_fence:
      callout_match = _CALLOUT_START_RE.match(line)
      if callout_match:
        indent = callout_match.group(1)
        kind = (callout_match.group(2) or callout_match.group(3)).lower()
        first_rest = callout_match.group(4)
        body_lines: list[str] = []
        if first_rest.strip():
          body_lines.append(first_rest)
        i += 1
        while i < len(lines):
          if _CALLOUT_START_RE.match(lines[i]):
            break
          bq_match = _BLOCKQUOTE_LINE_RE.match(lines[i])
          if not bq_match:
            break
          body_lines.append(bq_match.group(1))
          i += 1
        while body_lines and not body_lines[0].strip():
          body_lines.pop(0)
        while body_lines and not body_lines[-1].strip():
          body_lines.pop()
        out.append(f"{indent}!!! {kind}")
        for bline in body_lines:
          out.append(f"{indent}    {bline}" if bline.strip() else "")
        continue

    out.append(line)
    i += 1

  return "\n".join(out)


def on_page_markdown(markdown: str, **kwargs) -> str:
  """Uncomments `<!--mkdocs:...-->` directives and converts callout blockquotes."""
  del kwargs
  markdown = _MKDOCS_COMMENT_RE.sub(r"\1", markdown)
  return _convert_blockquote_callouts(markdown)
