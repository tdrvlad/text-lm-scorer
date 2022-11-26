from lm_scorer import LMScorer


def pretty_print(data):
    for actual_token, actual_token_p, best_token, best_token_p in data:
        print(f'({actual_token:7} ; {actual_token_p:.5f}) | ({best_token:7} ; {best_token_p:.5f})')


if __name__ == '__main__':
    lm_scorer = LMScorer()
    result = lm_scorer('Today is a nice day. Where shall we go?')
    pretty_print(result)
