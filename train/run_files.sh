# for i in {1..5}
# do
#     echo "run baseline.py VIT"
#     python baseline.py VIT 512
#     python ../test_inference/baseline.py VIT 512
#     sleep 120
# done

# noun
# echo "run with_noun.py RN101 512"
# python with_noun.py RN101 512
# python ../test_inference/with_noun.py RN101 512
# sleep 120

# echo "run with_noun.py VIT 512"
# python with_noun.py VIT 512
# python ../test_inference/with_noun.py VIT 512
# sleep 120

# for i in {1..5}
# do
#     echo "run with_noun.py VIT"
#     python with_noun.py VIT 512
#     python ../test_inference/with_noun.py VIT 512
#     sleep 120
# done

# echo "run with_noun.py VIT"
# python with_noun.py VIT 512
# python ../test_inference/with_noun.py VIT 512
# sleep 120

# # verb
# echo "run with_verb.py RN101 512"
# python with_verb.py RN101 512
# python ../test_inference/with_verb.py RN101 512
# sleep 120

# echo "run with_verb.py VIT 512"
# python with_verb.py VIT 512
# python ../test_inference/with_verb.py VIT 512
# sleep 120

# for i in {1..5}
# do
#     echo "run with_verb.py 512"
#     python with_verb.py VIT 512
#     python ../test_inference/with_verb.py VIT 512
#     sleep 120
# done

# # integrated
# echo "run integrated.py RN101 512"
# python integrated.py RN101 512
# python ../test_inference/integrated.py RN101 512
# sleep 120

# echo "run integrated.py VIT 512"
# python integrated.py VIT 512
# python ../test_inference/integrated.py VIT 512
# sleep 120

for i in {1..5}
do
    echo "run integrated.py VIT"
    python integrated.py VIT 512
    python ../test_inference/integrated.py VIT 512
    sleep 120
done