import pickle

dev_context_dict = pickle.load(open("dev_context_dict.p", "rb"))
dev_question_dict = pickle.load(open("dev_question_dict.p", "rb"))
dev_answer_dict = pickle.load(open("dev_answer_dict.p", "rb"))

print('below shown the context dict')
for c in list(dev_context_dict.keys())[:10]:
    print(c, ':', dev_context_dict[c])

print('below shown the question dict')
for q in list(dev_question_dict.keys())[:40]:
    print(q, ':', dev_question_dict[q])

print('below shown the answer dict')
for a in list(dev_answer_dict.keys())[:40]:
    print(a, ':', dev_answer_dict[a])

