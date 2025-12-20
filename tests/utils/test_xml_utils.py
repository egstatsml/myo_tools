"""
Copyright (c) 2025 MyoLab, Inc.

Released under the MyoLab Non-Commercial Scientific Research License
on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND,
either express or implied.

You may not use this file except in compliance with the License.
See the LICENSE file for governing permissions and limitations.
"""

# Summary: Tests for xml_utils

import xml.etree.ElementTree as ET
from unittest.mock import mock_open, patch

from myo_tools.utils.file_ops.xml_utils import (
    get_attribute,
    load_xml,
    read_file,
    remove_element_or_attribute,
    save_xml_to_file,
    update_or_set_attribute,
)

# Sample XML for testing
sample_xml = """<root>
    <child1 attribute1="value1" />
    <child2 attribute2="value2" />
</root>"""


def test_read_file():
    with patch("builtins.open", mock_open(read_data="file content")):
        content = read_file("dummy_path.txt")
        assert content == "file content"


def test_update_or_set_attribute():
    root = ET.fromstring(sample_xml)
    updated_xml = update_or_set_attribute(root, "child1", "attribute1", "new_value1")
    assert updated_xml.find("child1").get("attribute1") == "new_value1"

    updated_xml = update_or_set_attribute(root, "child3", "attribute3", "value3")
    assert updated_xml.find("child3").get("attribute3") == "value3"


def test_get_attribute():
    root = ET.fromstring(sample_xml)
    attribute_value = get_attribute(root, "child1", "attribute1")
    assert attribute_value == "value1"

    attribute_value = get_attribute(root, "child3", "attribute3")
    assert attribute_value is None


def test_remove_element_or_attribute():
    root = ET.fromstring(sample_xml)
    updated_xml = remove_element_or_attribute(root, "child2")
    assert updated_xml.find("child2") is None


def test_load_xml():
    with patch("xml.etree.ElementTree.parse") as mock_parse:
        mock_parse.return_value.getroot.return_value = ET.fromstring(sample_xml)
        root = load_xml("dummy_path.xml")
        assert root.tag == "root"
        assert root.find("child1").get("attribute1") == "value1"


def test_save_xml_to_file():
    root = ET.fromstring(sample_xml)
    with patch("builtins.open", mock_open()) as mock_file:
        save_xml_to_file(root, "dummy_output.xml")
        mock_file().write.assert_called_once()
        written_content = mock_file().write.call_args[0][0]
        assert "<root>" in written_content
        assert '<child1 attribute1="value1"/>' in written_content
        assert '<child2 attribute2="value2"/>' in written_content
