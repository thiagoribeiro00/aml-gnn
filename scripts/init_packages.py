import os

# Create empty __init__.py files in all directories to make them packages
directories = [
    'src',
    'src/domain',
    'src/use_cases',
    'src/adapters',
    'src/infrastructure',
    'src/models'
]

for d in directories:
    with open(os.path.join(d, '__init__.py'), 'w') as f:
        pass
