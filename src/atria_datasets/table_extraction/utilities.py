import json
import xml.etree.ElementTree as ET

from atria_core.types.generic.annotated_object import AnnotatedObject
from atria_core.types.generic.bounding_box import BoundingBox
from atria_core.types.generic.label import Label


def read_pascal_voc(
    xml_file: str, labels: list[str]
) -> tuple[list[BoundingBox], list[Label]]:
    tree = ET.parse(xml_file)
    root = tree.getroot()
    annotated_object = []
    for object_ in root.iter("object"):
        ymin, xmin, ymax, xmax = None, None, None, None
        label = object_.find("name").text
        try:
            label = int(label)
        except:
            label = labels.index(label)
        for box in object_.findall("bndbox"):
            ymin = float(box.find("ymin").text)
            xmin = float(box.find("xmin").text)
            ymax = float(box.find("ymax").text)
            xmax = float(box.find("xmax").text)
        annotated_object.append(
            AnnotatedObject(
                bbox=BoundingBox(value=[xmin, ymin, xmax, ymax]),
                label=Label(value=label, name=labels[label]),
            )
        )
    return annotated_object


def read_words_json(words_file: str):
    """
    Reads a JSON file containing words and their bounding boxes.

    Args:
        words_file (str): Path to the JSON file.

    Returns:
        List[AnnotatedObject]: A list of AnnotatedObject instances representing the words and their bounding boxes.
    """
    with open(words_file) as f:
        data = json.load(f)
    words = []
    word_bboxes = []
    for word in data:
        word_bboxes.append(BoundingBox(value=word["bbox"]))
        words.append(word["text"])
    return words, word_bboxes
