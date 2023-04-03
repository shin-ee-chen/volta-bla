import json

if __name__ == '__main__':
    input_json_path = "/root/autodl-tmp/datasets/FOIL/foil_debug.json"
    output_json_file = "/root/autodl-tmp/datasets/FOIL/annotations_debug/valid.json"
    
    procesed_data = {}
    with open(input_json_path) as f:
        captions = json.load(f)
        
    for caption_id, item in captions.items():
        img_id = item["img_name"][:-4]
        procesed_data[img_id] = item["captions"]
    
    json.dump(procesed_data, open(output_json_file, "w"), indent=2)