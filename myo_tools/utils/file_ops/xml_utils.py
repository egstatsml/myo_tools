"""
Copyright (c) 2026 MyoLab, Inc.

Released under the MyoLab Non-Commercial Scientific Research License
on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND,
either express or implied.

You may not use this file except in compliance with the License.
See the LICENSE file for governing permissions and limitations.
"""

# Summary: XML utilities

import xml.etree.ElementTree as ET

from lxml import etree


def load_markerset(markerset_handle: str | ET.Element):
    """
    Load the markerset from a file path or parse it from an XML element.
    Args:
        markerset_handle (str | ET.Element): The markerset definition, either as a file path or an XML element.
    Returns:
        markerset (ET.Element): The root element of the parsed markerset XML.
    """
    if isinstance(markerset_handle, str):
        if markerset_handle.endswith(".xml"):
            markerset = load_xml(markerset_handle)
        elif markerset_handle.strip().startswith("<markers"):
            markerset = ET.fromstring(markerset_handle)
        else:
            raise ValueError("Invalid markerset_handle format.")
    elif isinstance(markerset_handle, ET.Element):
        markerset = markerset_handle
    else:
        raise ValueError("Invalid markerset_handle type.")
    return markerset


def read_file(file_name):
    """
    Reads the contents of a file.

    Args:
        file_name (str): The name of the file to read.

    Returns:
        Contents of the file (str)
    """
    with open(file_name, mode="r") as f:
        return f.read()


def update_or_set_attribute(
    xml: ET.Element, element_name: str, attribute_name: str, attribute_value: str
):
    """
    Updates or sets the attribute value of an XML element.

    Args:
        xml (ET.Element): The XML element to update or set the attribute for.
        element_name (str): The name of the XML element.
        attribute_name (str): The name of the attribute to update or set.
        attribute_value (str): The value to update or set for the attribute.

    Returns:
        ET.Element: The updated XML element.

    """
    # show warning if multiple element_name elements are found
    if len(xml.findall(element_name)) > 1:
        print(
            "Warning: multiple %s elements detected. Updating only the first."
            % element_name
        )
    # Check if the element exists
    element = xml.find(element_name)
    if element is not None:
        # Update the existing attribute value
        element.set(attribute_name, attribute_value)
    else:
        # Create the element if it doesn't exist
        new_element = ET.Element(element_name)
        new_element.set(attribute_name, attribute_value)
        xml.append(new_element)
    return xml


def get_attribute(xml: ET.Element, element_name: str, attribute_name: str):
    """
    Get attribute value of an XML element.

    Args:
        xml (ET.Element): The XML element to update or set the attribute for.
        element_name (str): The name of the XML element.
        attribute_name (str): The name of the attribute to update or set.

            Returns:
        attribute_value

    """
    # show warning if multiple element_name elements are found
    if len(xml.findall(element_name)) > 1:
        print(
            "Warning: multiple %s elements detected. Getting only the first."
            % element_name
        )
    # Check if the element exists
    element = xml.find(element_name)
    if element is None:
        return None
    else:
        return element.get(attribute_name)


def remove_element_or_attribute(
    xml: ET.Element, element_name: str, attribute_name: str = None
):
    """
    Updates or sets the attribute value of an XML element.

    Args:
        xml (ET.Element): The XML element to update or set the attribute for.
        element_name (str): The name of the XML element.
        attribute_name (str): The name of the attribute to update or set.
        attribute_value (str): The value to update or set for the attribute.

    Returns:
        ET.Element: The updated XML element.

    """
    # show warning if multiple element_name elements are found
    if len(xml.findall(element_name)) > 1:
        print(
            "Warning: multiple %s elements detected. Updating only the first."
            % element_name
        )
    # Check if the element exists
    element = xml.findall(element_name)
    if len(element) == 0:
        return xml
    else:
        element = element[0]
        # Create the element if it doesn't exist
        if attribute_name is None:
            xml.remove(element)
            return xml
        if attribute_name in element.attrib:
            element.remove(attribute_name)
    return xml


def load_xml(file_path):
    """
    Load an XML file and return the root element.

    Parameters:
    file_path (str): The path to the XML file.

    Returns:
    xml.etree.ElementTree.Element: The root element of the XML file.
    """
    tree = ET.parse(file_path)
    root = tree.getroot()
    return root


def save_xml_to_file(xml: ET.Element, output_path: str):
    """
    Save an XML element to a file.

    Args:
        xml (Element): The XML element to be saved.
        output_path (str): The path to the output file.

    Returns:
        None
    """
    xml_str = ET.tostring(xml, encoding="utf-8").decode("utf-8")
    # etree_xml = etree.XML(xml_str, etree.XMLParser(remove_blank_text=True))
    etree_xml = etree.fromstring(xml_str)
    etree_xml_str = etree.tostring(etree_xml, pretty_print=True).decode("utf-8")
    with open(output_path, "w") as f:
        f.write(etree_xml_str)
