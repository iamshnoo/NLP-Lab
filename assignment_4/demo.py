from process import ProcessData
from unigram import UniGram


if __name__ == "__main__":
    x = ProcessData("demo_data.txt", 1)
    x.modify_data()
    train_data = x.get_lines()
    ngram = UniGram()
    unigram_counts = ngram.train(train_data)
    print("-" * 80)
    print("Unigram : ", unigram_counts)

    # print("Modified Unigram : ", unigram_counts)
    N = sum(unigram_counts.values())

    list_of_word_unit = list(unigram_counts.keys())
    word_unit_counts = unigram_counts
    total_number_of_word_unit = len(list(unigram_counts.keys()))
    bucket = {}
    for word_unit in word_unit_counts.items():
        value = word_unit[1]

        if value not in bucket:
            bucket[value] = 1
        else:
            bucket[value] += 1
    bucket_list = sorted(bucket.items(), key=lambda t: t[0])
    print("-" * 80)
    print("Nc Buckets : ", bucket)
    print("-" * 80)
    print("Sorted Nc Buckets : ", bucket_list)

    # N1 / N
    zero_occurence_prob = bucket_list[0][1] / N
    print("-" * 80)
    print("N1 : ", bucket_list[0][1])
    print("N :", N)
    print("Probability mass for zero occurences : ", zero_occurence_prob)

    last_item = bucket_list[len(bucket_list) - 1][0]
    for x in range(1, last_item):
        if x not in bucket:
            bucket[x] = 0
    bucket_list = sorted(bucket.items(), key=lambda t: t[0])

    print("-" * 80)
    print("Modified Nc Buckets : ", bucket)
    print("-" * 80)
    print("Sorted modified Nc Buckets : ", bucket_list)
    print("-" * 80)

    c_star = {}
    p_star = {}

    for c, nc in bucket_list:
        if nc == 0:
            c_star[c] = 0
            p_star[c] = 0
        else:
            if c == last_item:
                c_star[c] = -1
                p_star[c] = -1
            else:
                c_star[c] = (c + 1) * (bucket_list[c + 1][1]) / nc
                p_star[c] = c_star[c] / N

    print("-" * 80)
    print("C* :", c_star)
    print("-" * 80)
    print("P* :", p_star)

    list_of_probabilities = {}
    list_of_counts = {}

    for word_unit in list_of_word_unit:
        list_of_probabilities[word_unit] = p_star[word_unit_counts[word_unit]]
        list_of_counts[word_unit] = c_star[word_unit_counts[word_unit]]

    print("-" * 80)
    print(list_of_probabilities)
    print("-" * 80)
    print(list_of_counts)

    unigram_good_turing = list_of_probabilities
    unigram_zero_occurence_prob = zero_occurence_prob
    unigram_good_turing_cstar = list_of_counts

    sentence = "trout"
    score = 1
    for word in sentence.split():
        if word in unigram_good_turing:
            score *= unigram_good_turing[word]
        else:
            score *= unigram_zero_occurence_prob

    print("-" * 80)
    print("Calculated score for trout : ", score)
    print("Expected score for trout : ", (2 / 3) / 18)

    sentence = "bass"
    score = 1
    for word in sentence.split():
        if word in unigram_good_turing:
            score *= unigram_good_turing[word]
        else:
            score *= unigram_zero_occurence_prob

    print("-" * 80)
    print("Calculated score for bass : ", score)
    print("Expected score for bass : ", (3 / 18))
    print("-" * 80)
