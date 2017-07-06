from mab import ArticleArms

arms = ArticleArms('yahoo_r6_20090501.db')

# print(arms.current_event)
# print(arms.current_article)

print(arms.current_pool_articles)
print(arms.current_arm_names)
print(arms.current_arm_index)

"""
for i in range(arms.current_pool_size):
    print('arm {}({}): {}'.format(i, arms.current_arm_names[i], arms.get_article_features(i)))
for i in range(arms.current_pool_size):
    print('arm {}({}): {}'.format(i, arms.current_arm_names[i], arms.get_hybrid_features(i)))
"""

for i in range(35228):
    if i % 5000 == 0:
        print(i)
    a, b = arms.next()
    if a != 0 or bool(b) > 0:
        print('eid: {}, num_add: {}, dels: {}'.format(i, a, b))

print(arms.current_pool_articles)
print(arms.current_arm_names)
print(arms.current_arm_index)

print([arms.get_reward(i) for i in range(arms.current_pool_size)])
