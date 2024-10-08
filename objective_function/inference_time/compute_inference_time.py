import shutil
import subprocess

def compute_inference_time(calib, root_path, build_path):
    _generate_nsnet2_mpq_file(calib, build_path)
    _copy_c_files_to_build(root_path, build_path)
    inference_time =_compile_c_and_run(build_path)
    return inference_time

def _compile_c_and_run(build_path):
    result = subprocess.run(['gcc', '-o', 'nsnet2', 'main.c', 'npy_parser.c', 'q_nsnet2.c', '-lm'], cwd=build_path, capture_output=True, text=True)
    print(result.returncode)
    execute = subprocess.run(['./nsnet2'], cwd=build_path, capture_output=True, text=True)
    stdout = execute.stdout
    inference_time = float(stdout)
    return inference_time

def _generate_nsnet2_mpq_file(calib, build_path):
    code = ""
    code += _start_header_file()
    for key in calib.keys():
        level_code = _generate_macros_qt_level(key, calib)
        code += level_code
    code += _end_header_file()
    
    with open(build_path + 'mpq.h', 'w') as file:
        file.write(code)

def _copy_c_files_to_build(root_path, build_path):
    shutil.copy(root_path + 'nsnet2_c/npy_parser.c', build_path + 'npy_parser.c')
    shutil.copy(root_path + 'nsnet2_c/npy_parser.h', build_path + 'npy_parser.h')
    shutil.copy(root_path + 'nsnet2_c/q_nsnet2.h', build_path + 'q_nsnet2.h')
    shutil.copy(root_path + 'nsnet2_c/q_nsnet2.c', build_path + 'q_nsnet2.c')
    shutil.copy(root_path + 'nsnet2_c/main.c', build_path + 'main.c')

def _generate_macros_qt_level(key, calib):
    prefix = calib[key].macro_prefix
    bitwidth = calib[key].bitwidth
    type_qt = _get_qt_type(bitwidth)
    clip = int(2 ** (bitwidth) - 1)
    scaling_factor = calib[key].S()
    zero_point = int(calib[key].Z())

    code = ""
    code += "#define " + prefix + "NBITS " + str(clip) + "\n"
    code += "#define " + prefix + "TYPE " + str(type_qt) + "\n"
    code += "#define " + prefix + "S (" + str(scaling_factor) + ")\n"
    code += "#define " + prefix + "Z (" + str(zero_point) + ")\n\n"

    return code


def _get_qt_type(val):
    if val == 8:
        return "uint8_t"
    elif val == 16:
        return "uint16_t"
    elif val == 32:
        return "uint32_t"
    else:
        raise Exception("invalid quantization type")

def _start_header_file():
    code = "#ifndef __MPQ___\n" + \
            "#define __MPQ__\n"
    code += "\n\n"
    return code

def _end_header_file():
    code = "\n\n"
    code += "#endif // __MPQ__"
    return code