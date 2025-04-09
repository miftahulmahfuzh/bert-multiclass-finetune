def extract_xml_content(raw_output, tag="comment"):
    """
    Extracts content from raw_output based on the specified tag.

    Args:
        raw_output (str): The raw string containing potential XML-like tags.
        tag (str): The tag to look for (default: "comment").

    Returns:
        str: The extracted content, or the original string if no tag is found.
    """
    # Construct the opening and closing tags
    opening_tag = f"<{tag}>"
    closing_tag = f"</{tag}>"

    # Case 1: Check if both opening and closing tags exist
    if opening_tag in raw_output and closing_tag in raw_output:
        start_idx = raw_output.index(opening_tag) + len(opening_tag)
        end_idx = raw_output.index(closing_tag)
        return raw_output[start_idx:end_idx].strip()

    # Case 2: Check if only opening tag exists, without closing tag
    elif opening_tag in raw_output:
        start_idx = raw_output.index(opening_tag) + len(opening_tag)
        return raw_output[start_idx:].strip()

    # If no tags are found, return the original string
    return raw_output.strip()

def remove_tag_content(raw_output, tag="think"):
    """
    Removes content within specified tags including the tags themselves from raw_output.

    Args:
        raw_output (str): The raw string containing potential XML-like tags.
        tag (str): The tag to remove along with its content (default: "think").

    Returns:
        str: The string with specified tags and their content removed.
    """
    # Construct the opening and closing tags
    opening_tag = f"<{tag}>"
    closing_tag = f"</{tag}>"

    # Keep processing while tags exist
    while opening_tag in raw_output and closing_tag in raw_output:
        start_idx = raw_output.index(opening_tag)
        end_idx = raw_output.index(closing_tag) + len(closing_tag)
        # Remove the tagged content
        raw_output = raw_output[:start_idx] + raw_output[end_idx:]

    return raw_output.strip()

if __name__=="__main__":
    # Example usage with your output
    raw_output = "<comment>BBRI dan BBNI emang kena guruh, tapi kalo udah terjun, pasti lebih baik lagi</comment>\n\nInput:\n\n$BTPB $BTPB3 kena henti dulu ya"
    result = extract_xml_content(raw_output, tag="comment")
    print(f"Extracted content: {result}")
