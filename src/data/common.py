import numpy as np
import re

CLASS_NAMES = ['Pizza', 'Shawarma', 'Sushi']

SHORT_COLUMN_NAMES = {
    'id': 'id',
    'q1': 'Q1: From a scale 1 to 5, how complex is it to make this food? (Where 1 is the most simple, and 5 is the most complex)',
    'q2': 'Q2: How many ingredients would you expect this food item to contain?',
    'q3': 'Q3: In what setting would you expect this food to be served? Please check all that apply',
    'q4': 'Q4: How much would you expect to pay for one serving of this food item?',
    'q5': 'Q5: What movie do you think of when thinking of this food item?',
    'q6': 'Q6: What drink would you pair with this food item?',
    'q7': 'Q7: When you think about this food item, who does it remind you of?',
    'q8': 'Q8: How much hot sauce would you add to this food item?',
    'label': 'Label'
}

INTEGER_MAP = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15, "sixteen": 16, "seventeen": 17,
    "eighteen": 18, "nineteen": 19, "twenty": 20, "twenty-one": 21, "twenty-two": 22, "twenty-three": 23, "twenty-four": 24,
    "twenty-five": 25, "twenty-six": 26, "twenty-seven": 27, "twenty-eight": 28, "twenty-nine": 29, "thirty": 30,
    "thirty-one": 31, "thirty-two": 32, "thirty-three": 33, "thirty-four": 34, "thirty-five": 35, "thirty-six": 36,
    "thirty-seven": 37, "thirty-eight": 38, "thirty-nine": 39, "forty": 40, "forty-one": 41, "forty-two": 42, "forty-three": 43,
    "forty-four": 44, "forty-five": 45, "forty-six": 46, "forty-seven": 47, "forty-eight": 48, "forty-nine": 49, "fifty": 50,
    "fifty-one": 51, "fifty-two": 52, "fifty-three": 53, "fifty-four": 54, "fifty-five": 55, "fifty-six": 56, "fifty-seven": 57,
    "fifty-eight": 58, "fifty-nine": 59, "sixty": 60, "sixty-one": 61, "sixty-two": 62, "sixty-three": 63, "sixty-four": 64,
    "sixty-five": 65, "sixty-six": 66, "sixty-seven": 67, "sixty-eight": 68, "sixty-nine": 69, "seventy": 70, "seventy-one": 71,
    "seventy-two": 72, "seventy-three": 73, "seventy-four": 74, "seventy-five": 75, "seventy-six": 76, "seventy-seven": 77,
    "seventy-eight": 78, "seventy-nine": 79, "eighty": 80, "eighty-one": 81, "eighty-two": 82, "eighty-three": 83,
    "eighty-four": 84, "eighty-five": 85, "eighty-six": 86, "eighty-seven": 87, "eighty-eight": 88, "eighty-nine": 89,
    "ninety": 90, "ninety-one": 91, "ninety-two": 92, "ninety-three": 93, "ninety-four": 94, "ninety-five": 95,
    "ninety-six": 96, "ninety-seven": 97, "ninety-eight": 98, "ninety-nine": 99, "one-hundred": 100, "hundred": 100
}

ENGLISH_STOP_WORDS = set([
    'a',
    'about',
    'above',
    'across',
    'after',
    'afterwards',
    'again',
    'against',
    'ain',
    'all',
    'almost',
    'alone',
    'along',
    'already',
    'also',
    'although',
    'always',
    'am',
    'among',
    'amongst',
    'amoungst',
    'amount',
    'an',
    'and',
    'another',
    'any',
    'anyhow',
    'anyone',
    'anything',
    'anyway',
    'anywhere',
    'are',
    'aren',
    'around',
    'as',
    'at',
    'back',
    'be',
    'became',
    'because',
    'become',
    'becomes',
    'becoming',
    'been',
    'before',
    'beforehand',
    'behind',
    'being',
    'below',
    'beside',
    'besides',
    'between',
    'beyond',
    'bill',
    'both',
    'bottom',
    'but',
    'by',
    'call',
    'can',
    'cannot',
    'cant',
    'co',
    'con',
    'could',
    'couldn',
    'couldnt',
    'cry',
    'd',
    'de',
    'describe',
    'detail',
    'did',
    'didn',
    'do',
    'does',
    'doesn',
    'doing',
    'don',
    'done',
    'down',
    'due',
    'during',
    'each',
    'eg',
    'either',
    'else',
    'elsewhere',
    'empty',
    'enough',
    'etc',
    'even',
    'ever',
    'every',
    'everyone',
    'everything',
    'everywhere',
    'except',
    'few',
    'fify',
    'fill',
    'find',
    'fire',
    'first',
    'for',
    'former',
    'formerly',
    'found',
    'from',
    'front',
    'full',
    'further',
    'get',
    'give',
    'go',
    'had',
    'hadn',
    'has',
    'hasn',
    'hasnt',
    'have',
    'haven',
    'having',
    'he',
    'hence',
    'her',
    'here',
    'hereafter',
    'hereby',
    'herein',
    'hereupon',
    'hers',
    'herself',
    'him',
    'himself',
    'his',
    'how',
    'however',
    'i',
    'ie',
    'if',
    'in',
    'indeed',
    'interest',
    'into',
    'is',
    'isn',
    'it',
    'its',
    'itself',
    'just',
    'keep',
    'last',
    'latter',
    'latterly',
    'least',
    'less',
    'll',
    'ltd',
    'm',
    'ma',
    'made',
    'many',
    'may',
    'me',
    'meanwhile',
    'might',
    'mightn',
    'mill',
    'mine',
    'more',
    'moreover',
    'most',
    'mostly',
    'move',
    'much',
    'must',
    'mustn',
    'my',
    'myself',
    'name',
    'namely',
    'needn',
    'neither',
    'never',
    'nevertheless',
    'next',
    'no',
    'nobody',
    'none',
    'noone',
    'nor',
    'not',
    'nothing',
    'now',
    'nowhere',
    'o',
    'of',
    'off',
    'often',
    'on',
    'once',
    'only',
    'onto',
    'or',
    'other',
    'others',
    'otherwise',
    'our',
    'ours',
    'ourselves',
    'out',
    'over',
    'own',
    'part',
    'per',
    'perhaps',
    'please',
    'put',
    'rather',
    're',
    's',
    'same',
    'see',
    'seem',
    'seemed',
    'seeming',
    'seems',
    'serious',
    'several',
    'shan',
    'she',
    'should',
    'shouldn',
    'show',
    'side',
    'since',
    'sincere',
    'so',
    'some',
    'somehow',
    'someone',
    'something',
    'sometime',
    'sometimes',
    'somewhere',
    'still',
    'such',
    'system',
    't',
    'take',
    'than',
    'that',
    'the',
    'their',
    'theirs',
    'them',
    'themselves',
    'then',
    'thence',
    'there',
    'thereafter',
    'thereby',
    'therefore',
    'therein',
    'thereupon',
    'these',
    'they',
    'thick',
    'thin',
    'third',
    'this',
    'those',
    'though',
    'through',
    'throughout',
    'thru',
    'thus',
    'to',
    'together',
    'too',
    'top',
    'toward',
    'towards',
    'u',
    'un',
    'under',
    'until',
    'up',
    'upon',
    'us',
    've',
    'very',
    'via',
    'was',
    'wasn',
    'we',
    'well',
    'were',
    'weren',
    'what',
    'whatever',
    'when',
    'whence',
    'whenever',
    'where',
    'whereafter',
    'whereas',
    'whereby',
    'wherein',
    'whereupon',
    'wherever',
    'whether',
    'which',
    'while',
    'whither',
    'who',
    'whoever',
    'whole',
    'whom',
    'whose',
    'why',
    'will',
    'with',
    'within',
    'without',
    'won',
    'would',
    'wouldn',
    'y',
    'yet',
    'you',
    'your',
    'yours',
    'yourself',
    'yourselves'
])

def extract_numbers(response):
    matches = re.findall(r"\d*\.?\d+", response)
    return [float(match) for match in matches]

# def extract_numbers(response):
#     # Only keep digits and .
#     cleaned = ''.join(char if char.isnumeric() or char == '.' else ' ' for char in response)

#     tokens = cleaned.split() # Split on spaces

#     numbers = []
#     for token in tokens:
#         if token.endswith('.'):
#             token = token[:-1]  # Remove period, if present
#         try:
#             numbers.append(float(token))
#         except ValueError:
#             pass  # Ignore non-numeric parts
#     return numbers

def extract_and_aggregate_numbers(response, aggregation=np.mean):
    """
    Extracts numbers from a response and aggregates them using the specified function.

    Parameters:
    - response (str): The input string to extract numbers from.
    - aggregation (function): Specifies the function used to aggregate the results (e.g., np.mean, max, min).
                              Defaults to np.mean.

    Returns:
    - float representing the aggregated result based on the numbers found in the response or None.
    """
    response = response.lower()
    # 1) Handle numbers present in response
    numbers = extract_numbers(response)
    if numbers:
        return aggregation(numbers)

    # No numbers present
    # 2) Handle words spelled out as numbers (e.g., twenty-six to 26)
    cleaned = ''.join(char if char.isalpha() or char == '-' else ' ' for char in response)
    tokens = cleaned.split()  # Split on spaces
    for token in tokens:
        if token in INTEGER_MAP:
            numbers.append(INTEGER_MAP[token])
    if numbers:
        return aggregation(numbers)

    # Log or handle cases where no numbers are found
    # print(f"No numbers found in response: {response}")
    return None

def extract_ingredient_count(response, aggregation=np.mean):
    response = response.lower()
    count = extract_and_aggregate_numbers(response, aggregation)
    if count is None:
        # Try to handle list of ingredients
        cleaned_response = response.replace("\n", ",").replace("and", ",")
        if ',' in cleaned_response:
            count = len( [token for token in cleaned_response.split(",") if token.strip()] )
        else:
            count = None
    return count

def clean_text(response, remove_stop_words=False):
    # 1) Convert to lowercase
    response = response.lower()

    # 2) Keep only letters and spaces
    # response = ''.join(char if char.isalnum() or char.isspace() else '' for char in response)  # Keep only letters, numbers, and spaces
    response = ''.join(char if char.isalpha() or char.isspace() else '' for char in response)  # Keep only letters and spaces

    # 3) Split into words, filter out stop words, and rejoin
    if remove_stop_words:
        words = response.split()  # Split into words
        filtered_words = [word for word in words if word not in ENGLISH_STOP_WORDS]  # Filter stop words
        response = ' '.join(filtered_words)  # Rejoin into a single string

    return response
