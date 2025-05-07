from gpt_handler import GPT_Handler
import re
import ast
import pprint
from typing import Union
import os
from main import scenario_chooser
from analysis_scripts.show_table import get_data, create_df

GOAL_WAS_ACHIEVED_COUNTER = 0
GOAL_WASNT_ACHIEVED_COUNTER = 0
PROMPT_WAS_FAILED_COUNTER = 0

def swap_xy_any(data: Union[str, list, tuple, dict]):
    # Если пришла строка — парсим
    if isinstance(data, str):
        data_str = data.strip()
        try:
            parsed = ast.literal_eval(data_str)
        except Exception as e:
            raise ValueError(f"Не удалось разобрать строку: {e}")
        swapped = swap_xy_any(parsed)
        return pprint.pformat(swapped, indent=2, width=80)

    # Если список строк шагов
    if isinstance(data, list) and all(isinstance(item, str) and '->' in item for item in data):
        def swap_in_step(step):
            coords = re.findall(r'\((\d+),\s*(\d+)\)', step)
            swapped = [f"({y}, {x})" for x, y in coords]
            return f"{swapped[0]} -> {swapped[1]}"
        return [swap_in_step(step) for step in data]

    # Если список/кортеж координат
    elif isinstance(data, (list, tuple)) and all(isinstance(item, (tuple, list)) and len(item) == 2 for item in data):
        swapped = [(y, x) for x, y in data]
        return tuple(swapped) if isinstance(data, tuple) else swapped

    # Если словарь с координатами-ключами
    elif isinstance(data, dict) and all(isinstance(k, tuple) and len(k) == 2 for k in data):
        return {(y, x): v for (x, y), v in data.items()}

    else:
        raise ValueError("Неподдерживаемый формат данных.")

def is_valid_path(input_str, walls, field_size):

    match = re.search(r"=\s*(\[.*\])", input_str.strip(), re.DOTALL)
    if not match:
        print("Ошибка: не удалось извлечь список шагов из строки.")
        return False

    try:
        list_of_steps = ast.literal_eval(match.group(1))
    except Exception as e:
        print(f"Ошибка при разборе списка шагов: {e}")
        return False
    
    list_of_steps = swap_xy_any(list_of_steps)

    rows, cols = field_size
    wall_set = set(walls)  # Ускорим проверку

    def parse_step(step):
        from_str, to_str = step.split("->")
        from_pos = tuple(map(int, from_str.strip(" ()").split(",")))
        to_pos = tuple(map(int, to_str.strip(" ()").split(",")))
        return from_pos, to_pos

    previous_to = None

    for i, step in enumerate(list_of_steps):
        try:
            from_pos, to_pos = parse_step(step)
        except:
            print("Ошиька в parse_step: ", step)
            return False

        # 1. Проверка, что to предыдущего совпадает с from текущего
        if previous_to is not None and from_pos != previous_to:
            print(f"Ошибка на шаге {i}: несогласованные координаты ({previous_to} != {from_pos})")
            return False

        # 2. Проверка границ поля
        for pos in [from_pos, to_pos]:
            if not (0 <= pos[0] < rows and 0 <= pos[1] < cols):
                print(f"Ошибка на шаге {i}: координаты {pos} выходят за границы поля")
                return False

        # 3. Проверка на стены
        if to_pos in wall_set:
            print(f"Ошибка на шаге {i}: наступание на стену {to_pos}")
            return False

        # 4. Проверка соседства (манхэттенское расстояние)
        dx = abs(from_pos[0] - to_pos[0])
        dy = abs(from_pos[1] - to_pos[1])
        if (dx + dy) > 1:
            print(f"Ошибка на шаге {i}: переход от {from_pos} к {to_pos} невозможен")
            return False

        previous_to = to_pos  # обновим

    return True

def get_max_progon_index(base_folder=None):
    max_index = -1

    if not os.path.isdir(base_folder):
        print(f"Папка '{base_folder}' не найдена.")
        return None

    for name in os.listdir(base_folder):
        path = os.path.join(base_folder, name)
        if os.path.isdir(path):
            match = re.fullmatch(r'progon_(\d+)', name)
            if match:
                index = int(match.group(1))
                max_index = max(max_index, index)

    if max_index == -1:
        print("Папки с шаблоном 'progon_i' не найдены.")
        return None

    return max_index

def read_prompt_from_file(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            prompt = file.read()
        return prompt
    except FileNotFoundError:
        print(f"Файл не найден: {file_path}")
        return None
    except Exception as e:
        print(f"Ошибка при чтении файла: {e}")
        return None

def find_last_list_of_steps(input_string):
    pattern = r'list_of_steps = \[.*?\]'

    matches = re.findall(pattern, input_string, flags=re.DOTALL)

    if matches:
        return matches[-1]
    else:
        return None
    
if __name__ == "__main__":

    model = "gpt-4"
    api_key = "YOUR_API_KEY"
    url = "https://openrouter.ai/api/v1/chat/completions"
    altruist_start_zone = [(2, 0), (2, 1), (2, 2), (3, 0), (3, 1), (3, 2)]
    patron_start_zone = [(0, 0), (0, 1), (0, 2)]
    client = GPT_Handler(model=model, api_key=api_key, url=url)

    file_path = 'prompt.txt'
    init_prompt_text = read_prompt_from_file(file_path)

    print("PASS_0")

    argv = ["scenario_num=3a"]
    scenario_chooser(argv)

    print("PASS_1")

    pickled_data = get_data(scenario_num="3a", agent_type="patron", progon_number=get_max_progon_index("3a"))
    dict_str = create_df(pickled_data, True) 

    argv = ["scenario_num=3b", "no_learn"]
    walls =  [(1, 0), (1, 1), (4, 1)]
    field_size = (5, 3)

    prompt_text_with_q = re.sub("PATRON_Q_TABLE", swap_xy_any(dict_str), init_prompt_text)
    prompt_text_with_q = re.sub("WALLS", str(swap_xy_any(walls)), prompt_text_with_q)

    for i in range(15):
        print("PASS_2")
        altruist_start = altruist_start_zone[0]
        patron_start = patron_start_zone[0]

        curr_prompt_text = re.sub("ALTRUIST_START", str(swap_xy_any([altruist_start])[0]), prompt_text_with_q)
        curr_prompt_text = re.sub("PATRON_START", str(swap_xy_any([patron_start])[0]), curr_prompt_text)

        print(curr_prompt_text)
        print(f"iteration number: {i}")

        gpt_request = client.gpt_request(curr_prompt_text)

        print(f"gpt_request: {gpt_request}")

        print("PASS_3")
        list_of_steps = find_last_list_of_steps(gpt_request)
        print(list_of_steps)
        if list_of_steps and is_valid_path(list_of_steps, walls, field_size):
            print(f"list_of_steps: {type(list_of_steps)} {list_of_steps}")

            match = re.search(r"=\s*(\[.*\])", list_of_steps.strip(), re.DOTALL)
            list_of_steps = ast.literal_eval(match.group(1))
            list_of_steps = str(swap_xy_any(list_of_steps))
            print(f"list_of_steps: {type(list_of_steps)} {list_of_steps}")
            total_reward, done = scenario_chooser(argv, list_of_steps, altruist_start, patron_start)
            if done:
                GOAL_WAS_ACHIEVED_COUNTER += 1
            else:
                GOAL_WASNT_ACHIEVED_COUNTER += 1

        else:
            PROMPT_WAS_FAILED_COUNTER += 1

    print("GOAL_WAS_ACHIEVED_COUNTER: ", GOAL_WAS_ACHIEVED_COUNTER)
    print("GOAL_WASNT_ACHIEVED_COUNTER: ", GOAL_WASNT_ACHIEVED_COUNTER)
    print("PROMPT_WAS_FAILED_COUNTER: ", PROMPT_WAS_FAILED_COUNTER)