import json
import requests

def download_template(url):
    response = requests.get(url)
    response.raise_for_status()
    return response.text

def replace_in_file(template, replacements):
    for placeholder, replacement in replacements.items():
        template = template.replace(placeholder, replacement)
    return template

def main():
    with open('model_def.json', 'r') as file:
        model_def = json.load(file)

    template_url = model_def["template_url"]
    template_content = download_template(template_url)

    output_content = replace_in_file(template_content, model_def["replacements"])
    output_filename = 'README.md'

    with open(output_filename, 'w') as output_file:
        output_file.write(output_content)
    print(f'Processed {output_filename}')

if __name__ == "__main__":
    main()
