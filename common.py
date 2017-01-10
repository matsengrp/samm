def mutate_string(begin_str, mutate_pos, mutate_value):
    """
    Mutate a string
    """
    return "%s%s%s" % (begin_str[:mutate_pos], mutate_value, begin_str[mutate_pos + 1:])
