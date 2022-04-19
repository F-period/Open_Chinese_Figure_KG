
def save_to_file(data: list, file_pt: str):
    with open(file_pt, 'w', encoding='utf8') as f:
        for quick_look in data:
            f.write(str(quick_look)+'\n')

