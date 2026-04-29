from google import genai
from google.genai import types
import os.path
import os
import sys

def fix_case(input_filename, output_filename, client, model="gemini-3.1-pro-preview"):
    instructions = "I ran a handwriting recognition neural network on a medieval English legal case, written in Latin.  Please try to correct the mistakes.  The neural network doesn't typically add or remove entire words, so try to keep a similar word count, if possible.  Also, preserve the line breaks at all costs.  If you see a very short line, don't delete it or merge it with the previous or next line, because it's likely an interlinear addition.  Keep in mind that this is medieval, not classical, Latin.  Please don't add punctuation unless necessary, because these documents usually don't have much punctuation.  Do not output anything other than the corrected transcription in PageXML format--no explanations, comments, or alternatives.  The output should be identical to the input PageXML file, except that the transcriptions are replaced."
    case_txt = ""
    for line in open(input_filename):
        case_txt += line
    num_lines = case_txt.strip().count("\n") + 1
    instructions = instructions.format(num_lines)

    response = client.models.generate_content(
        model=model,
        config=types.GenerateContentConfig(
            system_instruction=instructions,
            temperature=1.0
        ),
        contents=case_txt
    )

    with open(output_filename, "w") as f:
        f.write(response.text + "\n")

print("Running Gemini...")
client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
fix_case(sys.argv[1], sys.argv[2], client)
