def getMaxSum(value_list):

    if len(value_list) == 0:
        return None

    sub_max_sum = 0
    sub_sum = 0

    for value in value_list:
        sub_sum = sub_sum + value
        if sub_sum < 0:
            sub_sum = 0
        if sub_sum > sub_max_sum:
            sub_max_sum = sub_sum

    if sub_max_sum == 0:
        sub_max_sum = value_list[0]
        for value in value_list[1:]:
            if value > sub_max_sum:
                sub_max_sum = value

    return sub_max_sum

def reverseWord(string):

    if len(string) == 0:
        return string

    word_set = []
    word_length = 0

    for i, strs in enumerate(string):
        if strs == ' ' or i == len(string)-1:
            word_set.append(string[i-word_length:i+1])
            word_length = 0
        else:
            word_length += 1

    word_set = [word_set[i] for i in range(len(word_set)-1, -1, -1)]

    string = " ".join(word_set)
    return string

if __name__ == "__main__":
    values = [1, -2, 3, 10, -4, 7, 2, -5]
    print(getMaxSum(values))
    values = [1, -2, 10, -4, 7, 2, -5]
    print(getMaxSum(values))
    teststr = 'I am a student.'
    print(reverseWord(teststr))
    print(reverseWord('Tested on real data, the algorithm not only provide the object orientation, but also more complete object compared with related work.'))

