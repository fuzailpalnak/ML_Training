from termcolor import colored


def colored_dual_string_print(print_string_1, print_string_2, color_string_1, color_string_2, attrs=None):
    print("[{}] - {}".format(colored("{}", color_string_1, attrs=attrs).format(print_string_1), colored(print_string_2,
                                                                                                        color_string_2)))


def colored_single_string_print(print_string_1, color_string_1, attrs=None):
    print("[{}]".format(colored("{}", color_string_1, attrs=attrs).format(print_string_1)))
