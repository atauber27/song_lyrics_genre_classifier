import torch
from transformers import pipeline,AutoModelForCausalLM, AutoTokenizer
prompts= ['''
Prompt:
Genre: Rap
Starting Lyric: "What they talkin' 'bout? They ain't talkin' 'bout nothin'"

Output:
What they talkin' 'bout? They ain't talkin' 'bout nothin'
''',
'''
Prompt:
Genre: Rock
Starting Lyric: "What they talkin' 'bout? They ain't talkin' 'bout nothin'"

Output:
What they talkin' 'bout? They ain't talkin' 'bout nothin'
''',
'''
Prompt:
Genre: Pop
Starting Lyric: "What they talkin' 'bout? They ain't talkin' 'bout nothin'"

Output:
What they talkin' 'bout? They ain't talkin' 'bout nothin'
''',
'''
Prompt:
Genre: R&B
Starting Lyric: "What they talkin' 'bout? They ain't talkin' 'bout nothin'"

Output:
What they talkin' 'bout? They ain't talkin' 'bout nothin'
''',
'''
Prompt:
Genre: Misc
Starting Lyric: "What they talkin' 'bout? They ain't talkin' 'bout nothin'"

Output:
What they talkin' 'bout? They ain't talkin' 'bout nothin'
''',
'''
Prompt:
Genre: Country
Starting Lyric: "What they talkin' 'bout? They ain't talkin' 'bout nothin'"

Output:
What they talkin' 'bout? They ain't talkin' 'bout nothin'
''','''
Prompt:
Genre: Rap
Starting Lyric: "Now he's thinkin' 'bout me every night, oh"

Output:
Now he's thinkin' 'bout me every night, oh''',
'''
Prompt:
Genre: Rock
Starting Lyric: "Now he's thinkin' 'bout me every night, oh"

Output:
Now he's thinkin' 'bout me every night, oh''',
'''
Prompt:
Genre: Pop
Starting Lyric: "Now he's thinkin' 'bout me every night, oh"

Output:
Now he's thinkin' 'bout me every night, oh''',
'''
Prompt:
Genre: R&B
Starting Lyric: "Now he's thinkin' 'bout me every night, oh"

Output:
Now he's thinkin' 'bout me every night, oh''',
'''
Prompt:
Genre: Misc
Starting Lyric: "Now he's thinkin' 'bout me every night, oh"

Output:
Now he's thinkin' 'bout me every night, oh''',
'''
Prompt:
Genre: Country
Starting Lyric: "Now he's thinkin' 'bout me every night, oh"

Output:
Now he's thinkin' 'bout me every night, oh''','''
Prompt:
Genre: Rap
Starting Lyric: "You're never gonna be alone, never from now on

Output:
You're never gonna be alone, never from now on''',
'''
Prompt:
Genre: Rock
Starting Lyric: "You're never gonna be alone, never from now on"

Output:
You're never gonna be alone, never from now on''',
'''
Prompt:
Genre: Pop
Starting Lyric: "You're never gonna be alone, never from now on"

Output:
You're never gonna be alone, never from now on''',
'''
Prompt:
Genre: R&B
Starting Lyric: "You're never gonna be alone, never from now on"

Output:
You're never gonna be alone, never from now on''',
'''
Prompt:
Genre: Misc
Starting Lyric: "You're never gonna be alone, never from now on"

Output:
You're never gonna be alone, never from now on''',
'''
Prompt:
Genre: Country
Starting Lyric: "You're never gonna be alone, never from now on"

Output:
You're never gonna be alone, never from now on''',
"""
Prompt:
Genre: Rap
Starting Lyric: "Catch me runnin' through the fire, ain't no lookin' back"

Output:
Catch me runnin' through the fire, ain't no lookin' back
""",
          """
Prompt:
Genre: Rock
Starting Lyric: "Catch me runnin' through the fire, ain't no lookin' back"

Output:
Catch me runnin' through the fire, ain't no lookin' back
""",
"""
Prompt:
Genre: Pop
Starting Lyric: "Catch me runnin' through the fire, ain't no lookin' back"

Output:
Catch me runnin' through the fire, ain't no lookin' back
""",
          """
Prompt:
Genre: R&B
Starting Lyric: "Catch me runnin' through the fire, ain't no lookin' back"

Output:
Catch me runnin' through the fire, ain't no lookin' back
""",
          """
Prompt:
Genre: Misc
Starting Lyric: "Catch me runnin' through the fire, ain't no lookin' back"

Output:
Catch me runnin' through the fire, ain't no lookin' back
""",
          """
Prompt:
Genre: Country
Starting Lyric: "Catch me runnin' through the fire, ain't no lookin' back"

Output:
Catch me runnin' through the fire, ain't no lookin' back
""",
          """
Prompt:
Genre: Rap
Starting Lyric: "Underneath the city lights, stories come alive"

Output:
Underneath the city lights, stories come alive
""",
          """
Prompt:
Genre: Rock
Starting Lyric: "Underneath the city lights, stories come alive"

Output:
Underneath the city lights, stories come alive
""",
          """
Prompt:
Genre: Pop
Starting Lyric: "Underneath the city lights, stories come alive"

Output:
Underneath the city lights, stories come alive
""",
          """
Prompt:
Genre: R&B
Starting Lyric: "Underneath the city lights, stories come alive"

Output:
Underneath the city lights, stories come alive
""",
          """
Prompt:
Genre: Misc
Starting Lyric: "Underneath the city lights, stories come alive"

Output:
Underneath the city lights, stories come alive
""",
          """
Prompt:
Genre: Country
Starting Lyric: "Underneath the city lights, stories come alive"

Output:
Underneath the city lights, stories come alive
""",
          """
Prompt:
Genre: Rap
Starting Lyric: "Love is a battlefield, can't fight it with a shield"

Output:
Love is a battlefield, can't fight it with a shield
""",
          """
Prompt:
Genre: Rock
Starting Lyric: "Love is a battlefield, can't fight it with a shield"

Output:
Love is a battlefield, can't fight it with a shield
""",
          """
Prompt:
Genre: Pop
Starting Lyric: "Love is a battlefield, can't fight it with a shield"

Output:
Love is a battlefield, can't fight it with a shield
""",
          """
Prompt:
Genre: R&B
Starting Lyric: "Love is a battlefield, can't fight it with a shield"

Output:
Love is a battlefield, can't fight it with a shield
""",
          """
Prompt:
Genre: Misc
Starting Lyric: "Love is a battlefield, can't fight it with a shield"

Output:
Love is a battlefield, can't fight it with a shield
""",
          """
Prompt:
Genre: Country
Starting Lyric: "Love is a battlefield, can't fight it with a shield"

Output:
Love is a battlefield, can't fight it with a shield
""",
          '''
Prompt:
Genre: Rap
Starting Lyric: "Dreams don't wait, gotta catch 'em while they're flyin'"

Output:
Dreams don't wait, gotta catch 'em while they're flyin'
''',
'''
Prompt:
Genre: Rock
Starting Lyric: "Dreams don't wait, gotta catch 'em while they're flyin'"

Output:
Dreams don't wait, gotta catch 'em while they're flyin'
''',
'''
Prompt:
Genre: Pop
Starting Lyric: "Dreams don't wait, gotta catch 'em while they're flyin'"

Output:
Dreams don't wait, gotta catch 'em while they're flyin'
''',
'''
Prompt:
Genre: R&B
Starting Lyric: "Dreams don't wait, gotta catch 'em while they're flyin'"

Output:
Dreams don't wait, gotta catch 'em while they're flyin'
''',
'''
Prompt:
Genre: Misc
Starting Lyric: "Dreams don't wait, gotta catch 'em while they're flyin'"

Output:
Dreams don't wait, gotta catch 'em while they're flyin'
''',
'''
Prompt:
Genre: Country
Starting Lyric: "Dreams don't wait, gotta catch 'em while they're flyin'"

Output:
Dreams don't wait, gotta catch 'em while they're flyin'
''',
'''
Prompt:
Genre: Rap
Starting Lyric: "On this journey, every step feels like the first"

Output:
On this journey, every step feels like the first
''',
'''
Prompt:
Genre: Rock
Starting Lyric: "On this journey, every step feels like the first"

Output:
On this journey, every step feels like the first
''',
'''
Prompt:
Genre: Pop
Starting Lyric: "On this journey, every step feels like the first"

Output:
On this journey, every step feels like the first
''',
'''
Prompt:
Genre: R&B
Starting Lyric: "On this journey, every step feels like the first"

Output:
On this journey, every step feels like the first
''',
'''
Prompt:
Genre: Misc
Starting Lyric: "On this journey, every step feels like the first"

Output:
On this journey, every step feels like the first
''',
'''
Prompt:
Genre: Country
Starting Lyric: "On this journey, every step feels like the first"

Output:
On this journey, every step feels like the first
''',
'''
Prompt:
Genre: Rap
Starting Lyric: "The rain keeps fallin', but the grind don't stop"

Output:
The rain keeps fallin', but the grind don't stop
''',
'''
Prompt:
Genre: Rock
Starting Lyric: "The rain keeps fallin', but the grind don't stop"

Output:
The rain keeps fallin', but the grind don't stop
''',
'''
Prompt:
Genre: Pop
Starting Lyric: "The rain keeps fallin', but the grind don't stop"

Output:
The rain keeps fallin', but the grind don't stop
''',
'''
Prompt:
Genre: R&B
Starting Lyric: "The rain keeps fallin', but the grind don't stop"

Output:
The rain keeps fallin', but the grind don't stop
''',
'''
Prompt:
Genre: Misc
Starting Lyric: "The rain keeps fallin', but the grind don't stop"

Output:
The rain keeps fallin', but the grind don't stop
''',
'''
Prompt:
Genre: Country
Starting Lyric: "The rain keeps fallin', but the grind don't stop"

Output:
The rain keeps fallin', but the grind don't stop
''',
'''
Prompt:
Genre: Rap
Starting Lyric: "The stars align, but I make my own fate"

Output:
The stars align, but I make my own fate
''',
'''
Prompt:
Genre: Rock
Starting Lyric: "The stars align, but I make my own fate"

Output:
The stars align, but I make my own fate
''',
'''
Prompt:
Genre: Pop
Starting Lyric: "The stars align, but I make my own fate"

Output:
The stars align, but I make my own fate
''',
'''
Prompt:
Genre: R&B
Starting Lyric: "The stars align, but I make my own fate"

Output:
The stars align, but I make my own fate
''',
'''
Prompt:
Genre: Misc
Starting Lyric: "The stars align, but I make my own fate"

Output:
The stars align, but I make my own fate
''',
'''
Prompt:
Genre: Country
Starting Lyric: "The stars align, but I make my own fate"

Output:
The stars align, but I make my own fate
''',
'''
Prompt:
Genre: Rap
Starting Lyric: "Life's a hustle, gotta stay on the grind"

Output:
Life's a hustle, gotta stay on the grind
''',
'''
Prompt:
Genre: Rock
Starting Lyric: "Life's a hustle, gotta stay on the grind"

Output:
Life's a hustle, gotta stay on the grind
''',
'''
Prompt:
Genre: Pop
Starting Lyric: "Life's a hustle, gotta stay on the grind"

Output:
Life's a hustle, gotta stay on the grind
''',
'''
Prompt:
Genre: R&B
Starting Lyric: "Life's a hustle, gotta stay on the grind"

Output:
Life's a hustle, gotta stay on the grind
''',
'''
Prompt:
Genre: Misc
Starting Lyric: "Life's a hustle, gotta stay on the grind"

Output:
Life's a hustle, gotta stay on the grind
''',
'''
Prompt:
Genre: Country
Starting Lyric: "Life's a hustle, gotta stay on the grind"

Output:
Life's a hustle, gotta stay on the grind
''',
'''
Prompt:
Genre: Rap
Starting Lyric: "Through the fire, I rise to the sky"

Output:
Through the fire, I rise to the sky
''',
'''
Prompt:
Genre: Rock
Starting Lyric: "Through the fire, I rise to the sky"

Output:
Through the fire, I rise to the sky
''',
'''
Prompt:
Genre: Pop
Starting Lyric: "Through the fire, I rise to the sky"

Output:
Through the fire, I rise to the sky
''',
'''
Prompt:
Genre: R&B
Starting Lyric: "Through the fire, I rise to the sky"

Output:
Through the fire, I rise to the sky
''',
'''
Prompt:
Genre: Misc
Starting Lyric: "Through the fire, I rise to the sky"

Output:
Through the fire, I rise to the sky
''',
'''
Prompt:
Genre: Country
Starting Lyric: "Through the fire, I rise to the sky"

Output:
Through the fire, I rise to the sky
''',
'''
Prompt:
Genre: Rap
Starting Lyric: "Every moment's a treasure, don't let it fade"

Output:
Every moment's a treasure, don't let it fade
''',
'''
Prompt:
Genre: Rock
Starting Lyric: "Every moment's a treasure, don't let it fade"

Output:
Every moment's a treasure, don't let it fade
''',
'''
Prompt:
Genre: Pop
Starting Lyric: "Every moment's a treasure, don't let it fade"

Output:
Every moment's a treasure, don't let it fade
''',
'''
Prompt:
Genre: R&B
Starting Lyric: "Every moment's a treasure, don't let it fade"

Output:
Every moment's a treasure, don't let it fade
''',
'''
Prompt:
Genre: Misc
Starting Lyric: "Every moment's a treasure, don't let it fade"

Output:
Every moment's a treasure, don't let it fade
''',
'''
Prompt:
Genre: Country
Starting Lyric: "Every moment's a treasure, don't let it fade"

Output:
Every moment's a treasure, don't let it fade
''',
'''
Prompt:
Genre: Rap
Starting Lyric: "Chasing dreams, running faster than the wind"

Output:
Chasing dreams, running faster than the wind
''',
'''
Prompt:
Genre: Rock
Starting Lyric: "Chasing dreams, running faster than the wind"

Output:
Chasing dreams, running faster than the wind
''',
'''
Prompt:
Genre: Pop
Starting Lyric: "Chasing dreams, running faster than the wind"

Output:
Chasing dreams, running faster than the wind
''',
'''
Prompt:
Genre: R&B
Starting Lyric: "Chasing dreams, running faster than the wind"

Output:
Chasing dreams, running faster than the wind
''',
'''
Prompt:
Genre: Misc
Starting Lyric: "Chasing dreams, running faster than the wind"

Output:
Chasing dreams, running faster than the wind
''',
'''
Prompt:
Genre: Country
Starting Lyric: "Chasing dreams, running faster than the wind"

Output:
Chasing dreams, running faster than the wind
''',
'''
Prompt:
Genre: Rap
Starting Lyric: "Under the city lights, the world comes alive"

Output:
Under the city lights, the world comes alive
''',
'''
Prompt:
Genre: Rock
Starting Lyric: "Under the city lights, the world comes alive"

Output:
Under the city lights, the world comes alive
''',
'''
Prompt:
Genre: Pop
Starting Lyric: "Under the city lights, the world comes alive"

Output:
Under the city lights, the world comes alive
''',
'''
Prompt:
Genre: R&B
Starting Lyric: "Under the city lights, the world comes alive"

Output:
Under the city lights, the world comes alive
''',
'''
Prompt:
Genre: Misc
Starting Lyric: "Under the city lights, the world comes alive"

Output:
Under the city lights, the world comes alive
''',
'''
Prompt:
Genre: Country
Starting Lyric: "Under the city lights, the world comes alive"

Output:
Under the city lights, the world comes alive
''',
'''
Prompt:
Genre: Rap
Starting Lyric: "The waves crash down, but I still stand tall"

Output:
The waves crash down, but I still stand tall
''',
'''
Prompt:
Genre: Rock
Starting Lyric: "The waves crash down, but I still stand tall"

Output:
The waves crash down, but I still stand tall
''',
'''
Prompt:
Genre: Pop
Starting Lyric: "The waves crash down, but I still stand tall"

Output:
The waves crash down, but I still stand tall
''',
'''
Prompt:
Genre: R&B
Starting Lyric: "The waves crash down, but I still stand tall"

Output:
The waves crash down, but I still stand tall
''',
'''
Prompt:
Genre: Misc
Starting Lyric: "The waves crash down, but I still stand tall"

Output:
The waves crash down, but I still stand tall
''',
'''
Prompt:
Genre: Country
Starting Lyric: "The waves crash down, but I still stand tall"

Output:
The waves crash down, but I still stand tall
''',
          '''
Prompt:
Genre: Rap
Starting Lyric: "Rise up from the ashes, I’m ready to fight"

Output:
Rise up from the ashes, I’m ready to fight
''',
'''
Prompt:
Genre: Rock
Starting Lyric: "Rise up from the ashes, I’m ready to fight"

Output:
Rise up from the ashes, I’m ready to fight
''',
'''
Prompt:
Genre: Pop
Starting Lyric: "Rise up from the ashes, I’m ready to fight"

Output:
Rise up from the ashes, I’m ready to fight
''',
'''
Prompt:
Genre: R&B
Starting Lyric: "Rise up from the ashes, I’m ready to fight"

Output:
Rise up from the ashes, I’m ready to fight
''',
'''
Prompt:
Genre: Misc
Starting Lyric: "Rise up from the ashes, I’m ready to fight"

Output:
Rise up from the ashes, I’m ready to fight
''',
'''
Prompt:
Genre: Country
Starting Lyric: "Rise up from the ashes, I’m ready to fight"

Output:
Rise up from the ashes, I’m ready to fight
''',
'''
Prompt:
Genre: Rap
Starting Lyric: "I see the stars align, guiding my way home"

Output:
I see the stars align, guiding my way home
''',
'''
Prompt:
Genre: Rock
Starting Lyric: "I see the stars align, guiding my way home"

Output:
I see the stars align, guiding my way home
''',
'''
Prompt:
Genre: Pop
Starting Lyric: "I see the stars align, guiding my way home"

Output:
I see the stars align, guiding my way home
''',
'''
Prompt:
Genre: R&B
Starting Lyric: "I see the stars align, guiding my way home"

Output:
I see the stars align, guiding my way home
''',
'''
Prompt:
Genre: Misc
Starting Lyric: "I see the stars align, guiding my way home"

Output:
I see the stars align, guiding my way home
''',
'''
Prompt:
Genre: Country
Starting Lyric: "I see the stars align, guiding my way home"

Output:
I see the stars align, guiding my way home
''',
'''
Prompt:
Genre: Rap
Starting Lyric: "Heartbeats echo, louder than the crowd"

Output:
Heartbeats echo, louder than the crowd
''',
'''
Prompt:
Genre: Rock
Starting Lyric: "Heartbeats echo, louder than the crowd"

Output:
Heartbeats echo, louder than the crowd
''',
'''
Prompt:
Genre: Pop
Starting Lyric: "Heartbeats echo, louder than the crowd"

Output:
Heartbeats echo, louder than the crowd
''',
'''
Prompt:
Genre: R&B
Starting Lyric: "Heartbeats echo, louder than the crowd"

Output:
Heartbeats echo, louder than the crowd
''',
'''
Prompt:
Genre: Misc
Starting Lyric: "Heartbeats echo, louder than the crowd"

Output:
Heartbeats echo, louder than the crowd
''',
'''
Prompt:
Genre: Country
Starting Lyric: "Heartbeats echo, louder than the crowd"

Output:
Heartbeats echo, louder than the crowd
''',
'''
Prompt:
Genre: Rap
Starting Lyric: "Under the neon lights, the future is ours"

Output:
Under the neon lights, the future is ours
''',
'''
Prompt:
Genre: Rock
Starting Lyric: "Under the neon lights, the future is ours"

Output:
Under the neon lights, the future is ours
''',
'''
Prompt:
Genre: Pop
Starting Lyric: "Under the neon lights, the future is ours"

Output:
Under the neon lights, the future is ours
''',
'''
Prompt:
Genre: R&B
Starting Lyric: "Under the neon lights, the future is ours"

Output:
Under the neon lights, the future is ours
''',
'''
Prompt:
Genre: Misc
Starting Lyric: "Under the neon lights, the future is ours"

Output:
Under the neon lights, the future is ours
''',
'''
Prompt:
Genre: Country
Starting Lyric: "Under the neon lights, the future is ours"

Output:
Under the neon lights, the future is ours
''',
'''
Prompt:
Genre: Rap
Starting Lyric: "My story’s untold, but the fire still burns"

Output:
My story’s untold, but the fire still burns
''',
'''
Prompt:
Genre: Rock
Starting Lyric: "My story’s untold, but the fire still burns"

Output:
My story’s untold, but the fire still burns
''',
'''
Prompt:
Genre: Pop
Starting Lyric: "My story’s untold, but the fire still burns"

Output:
My story’s untold, but the fire still burns
''',
'''
Prompt:
Genre: R&B
Starting Lyric: "My story’s untold, but the fire still burns"

Output:
My story’s untold, but the fire still burns
''',
'''
Prompt:
Genre: Misc
Starting Lyric: "My story’s untold, but the fire still burns"

Output:
My story’s untold, but the fire still burns
''',
'''
Prompt:
Genre: Country
Starting Lyric: "My story’s untold, but the fire still burns"

Output:
My story’s untold, but the fire still burns
''',
'''
Prompt:
Genre: Rap
Starting Lyric: "From the hills to the valleys, I’m making my way"

Output:
From the hills to the valleys, I’m making my way
''',
'''
Prompt:
Genre: Rock
Starting Lyric: "From the hills to the valleys, I’m making my way"

Output:
From the hills to the valleys, I’m making my way
''',
'''
Prompt:
Genre: Pop
Starting Lyric: "From the hills to the valleys, I’m making my way"

Output:
From the hills to the valleys, I’m making my way
''',
'''
Prompt:
Genre: R&B
Starting Lyric: "From the hills to the valleys, I’m making my way"

Output:
From the hills to the valleys, I’m making my way
''',
'''
Prompt:
Genre: Misc
Starting Lyric: "From the hills to the valleys, I’m making my way"

Output:
From the hills to the valleys, I’m making my way
''',
'''
Prompt:
Genre: Country
Starting Lyric: "From the hills to the valleys, I’m making my way"

Output:
From the hills to the valleys, I’m making my way
''',
'''
Prompt:
Genre: Rap
Starting Lyric: "The shadows fade when the spotlight shines"

Output:
The shadows fade when the spotlight shines
''',
'''
Prompt:
Genre: Rock
Starting Lyric: "The shadows fade when the spotlight shines"

Output:
The shadows fade when the spotlight shines
''',
'''
Prompt:
Genre: Pop
Starting Lyric: "The shadows fade when the spotlight shines"

Output:
The shadows fade when the spotlight shines
''',
'''
Prompt:
Genre: R&B
Starting Lyric: "The shadows fade when the spotlight shines"

Output:
The shadows fade when the spotlight shines
''',
'''
Prompt:
Genre: Misc
Starting Lyric: "The shadows fade when the spotlight shines"

Output:
The shadows fade when the spotlight shines
''',
'''
Prompt:
Genre: Country
Starting Lyric: "The shadows fade when the spotlight shines"

Output:
The shadows fade when the spotlight shines
''',
'''
Prompt:
Genre: Rap
Starting Lyric: "The grind don’t stop, chasing dreams every night"

Output:
The grind don’t stop, chasing dreams every night
''',
'''
Prompt:
Genre: Rock
Starting Lyric: "The grind don’t stop, chasing dreams every night"

Output:
The grind don’t stop, chasing dreams every night
''',
'''
Prompt:
Genre: Pop
Starting Lyric: "The grind don’t stop, chasing dreams every night"

Output:
The grind don’t stop, chasing dreams every night
''',
'''
Prompt:
Genre: R&B
Starting Lyric: "The grind don’t stop, chasing dreams every night"

Output:
The grind don’t stop, chasing dreams every night
''',
'''
Prompt:
Genre: Misc
Starting Lyric: "The grind don’t stop, chasing dreams every night"

Output:
The grind don’t stop, chasing dreams every night
''',
'''
Prompt:
Genre: Country
Starting Lyric: "The grind don’t stop, chasing dreams every night"

Output:
The grind don’t stop, chasing dreams every night
''',
'''
Prompt:
Genre: Rap
Starting Lyric: "Every scar I’ve earned tells a story of strength"

Output:
Every scar I’ve earned tells a story of strength
''',
'''
Prompt:
Genre: Rock
Starting Lyric: "Every scar I’ve earned tells a story of strength"

Output:
Every scar I’ve earned tells a story of strength
''',
'''
Prompt:
Genre: Pop
Starting Lyric: "Every scar I’ve earned tells a story of strength"

Output:
Every scar I’ve earned tells a story of strength
''',
'''
Prompt:
Genre: R&B
Starting Lyric: "Every scar I’ve earned tells a story of strength"

Output:
Every scar I’ve earned tells a story of strength
''',
'''
Prompt:
Genre: Misc
Starting Lyric: "Every scar I’ve earned tells a story of strength"

Output:
Every scar I’ve earned tells a story of strength
''',
'''
Prompt:
Genre: Country
Starting Lyric: "Every scar I’ve earned tells a story of strength"

Output:
Every scar I’ve earned tells a story of strength
''',
'''
Prompt:
Genre: Rap
Starting Lyric: "Through the struggle, I’ve found my own voice"

Output:
Through the struggle, I’ve found my own voice
''',
'''
Prompt:
Genre: Rock
Starting Lyric: "Through the struggle, I’ve found my own voice"

Output:
Through the struggle, I’ve found my own voice
''',
'''
Prompt:
Genre: Pop
Starting Lyric: "Through the struggle, I’ve found my own voice"

Output:
Through the struggle, I’ve found my own voice
''',
'''
Prompt:
Genre: R&B
Starting Lyric: "Through the struggle, I’ve found my own voice"

Output:
Through the struggle, I’ve found my own voice
''',
'''
Prompt:
Genre: Misc
Starting Lyric: "Through the struggle, I’ve found my own voice"

Output:
Through the struggle, I’ve found my own voice
''',
'''
Prompt:
Genre: Country
Starting Lyric: "Through the struggle, I’ve found my own voice"

Output:
Through the struggle, I’ve found my own voice
''',
'''
Prompt:
Genre: Rap
Starting Lyric: "Walking down roads I’ve never seen before"

Output:
Walking down roads I’ve never seen before
''',
'''
Prompt:
Genre: Rock
Starting Lyric: "Walking down roads I’ve never seen before"

Output:
Walking down roads I’ve never seen before
''',
'''
Prompt:
Genre: Pop
Starting Lyric: "Walking down roads I’ve never seen before"

Output:
Walking down roads I’ve never seen before
''',
'''
Prompt:
Genre: R&B
Starting Lyric: "Walking down roads I’ve never seen before"

Output:
Walking down roads I’ve never seen before
''',
'''
Prompt:
Genre: Misc
Starting Lyric: "Walking down roads I’ve never seen before"

Output:
Walking down roads I’ve never seen before
''',
'''
Prompt:
Genre: Country
Starting Lyric: "Walking down roads I’ve never seen before"

Output:
Walking down roads I’ve never seen before
''',
'''
Prompt:
Genre: Rap
Starting Lyric: "Breaking the chains, I’m finally free"

Output:
Breaking the chains, I’m finally free
''',
'''
Prompt:
Genre: Rock
Starting Lyric: "Breaking the chains, I’m finally free"

Output:
Breaking the chains, I’m finally free
''',
'''
Prompt:
Genre: Pop
Starting Lyric: "Breaking the chains, I’m finally free"

Output:
Breaking the chains, I’m finally free
''',
'''
Prompt:
Genre: R&B
Starting Lyric: "Breaking the chains, I’m finally free"

Output:
Breaking the chains, I’m finally free
''',
'''
Prompt:
Genre: Misc
Starting Lyric: "Breaking the chains, I’m finally free"

Output:
Breaking the chains, I’m finally free
''',
'''
Prompt:
Genre: Country
Starting Lyric: "Breaking the chains, I’m finally free"

Output:
Breaking the chains, I’m finally free
''',
'''
Prompt:
Genre: Rap
Starting Lyric: "Caught in the storm, but I’ll never fall"

Output:
Caught in the storm, but I’ll never fall
''',
'''
Prompt:
Genre: Rock
Starting Lyric: "Caught in the storm, but I’ll never fall"

Output:
Caught in the storm, but I’ll never fall
''',
'''
Prompt:
Genre: Pop
Starting Lyric: "Caught in the storm, but I’ll never fall"

Output:
Caught in the storm, but I’ll never fall
''',
'''
Prompt:
Genre: R&B
Starting Lyric: "Caught in the storm, but I’ll never fall"

Output:
Caught in the storm, but I’ll never fall
''',
'''
Prompt:
Genre: Misc
Starting Lyric: "Caught in the storm, but I’ll never fall"

Output:
Caught in the storm, but I’ll never fall
''',
'''
Prompt:
Genre: Country
Starting Lyric: "Caught in the storm, but I’ll never fall"

Output:
Caught in the storm, but I’ll never fall
''',
'''
Prompt:
Genre: Rap
Starting Lyric: "Running through the fire, I’ve got no fear"

Output:
Running through the fire, I’ve got no fear
''',
'''
Prompt:
Genre: Rock
Starting Lyric: "Running through the fire, I’ve got no fear"

Output:
Running through the fire, I’ve got no fear
''',
'''
Prompt:
Genre: Pop
Starting Lyric: "Running through the fire, I’ve got no fear"

Output:
Running through the fire, I’ve got no fear
''',
'''
Prompt:
Genre: R&B
Starting Lyric: "Running through the fire, I’ve got no fear"

Output:
Running through the fire, I’ve got no fear
''',
'''
Prompt:
Genre: Misc
Starting Lyric: "Running through the fire, I’ve got no fear"

Output:
Running through the fire, I’ve got no fear
''',
'''
Prompt:
Genre: Country
Starting Lyric: "Running through the fire, I’ve got no fear"

Output:
Running through the fire, I’ve got no fear
''',
'''
Prompt:
Genre: Rap
Starting Lyric: "These streets whisper secrets I’ve yet to learn"

Output:
These streets whisper secrets I’ve yet to learn
''',
'''
Prompt:
Genre: Rock
Starting Lyric: "These streets whisper secrets I’ve yet to learn"

Output:
These streets whisper secrets I’ve yet to learn
''',
'''
Prompt:
Genre: Pop
Starting Lyric: "These streets whisper secrets I’ve yet to learn"

Output:
These streets whisper secrets I’ve yet to learn
''',
'''
Prompt:
Genre: R&B
Starting Lyric: "These streets whisper secrets I’ve yet to learn"

Output:
These streets whisper secrets I’ve yet to learn
''',
'''
Prompt:
Genre: Misc
Starting Lyric: "These streets whisper secrets I’ve yet to learn"

Output:
These streets whisper secrets I’ve yet to learn
''',
'''
Prompt:
Genre: Country
Starting Lyric: "These streets whisper secrets I’ve yet to learn"

Output:
These streets whisper secrets I’ve yet to learn
''',
'''
Prompt:
Genre: Rap
Starting Lyric: "I’ve been waiting for this moment all my life"

Output:
I’ve been waiting for this moment all my life
''',
'''
Prompt:
Genre: Rock
Starting Lyric: "I’ve been waiting for this moment all my life"

Output:
I’ve been waiting for this moment all my life
''',
'''
Prompt:
Genre: Pop
Starting Lyric: "I’ve been waiting for this moment all my life"

Output:
I’ve been waiting for this moment all my life
''',
'''
Prompt:
Genre: R&B
Starting Lyric: "I’ve been waiting for this moment all my life"

Output:
I’ve been waiting for this moment all my life
''',
'''
Prompt:
Genre: Misc
Starting Lyric: "I’ve been waiting for this moment all my life"

Output:
I’ve been waiting for this moment all my life
''',
'''
Prompt:
Genre: Country
Starting Lyric: "I’ve been waiting for this moment all my life"

Output:
I’ve been waiting for this moment all my life
''',
'''
Prompt:
Genre: Rap
Starting Lyric: "Turning my pain into something I can use"

Output:
Turning my pain into something I can use
''',
'''
Prompt:
Genre: Rock
Starting Lyric: "Turning my pain into something I can use"

Output:
Turning my pain into something I can use
''',
'''
Prompt:
Genre: Pop
Starting Lyric: "Turning my pain into something I can use"

Output:
Turning my pain into something I can use
''',
'''
Prompt:
Genre: R&B
Starting Lyric: "Turning my pain into something I can use"

Output:
Turning my pain into something I can use
''',
'''
Prompt:
Genre: Misc
Starting Lyric: "Turning my pain into something I can use"

Output:
Turning my pain into something I can use
''',
'''
Prompt:
Genre: Country
Starting Lyric: "Turning my pain into something I can use"

Output:
Turning my pain into something I can use
'''
]
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("fine_tuned_gpt2").to("cuda")  # Move model to GPU
tokenizer = AutoTokenizer.from_pretrained("fine_tuned_gpt2", padding_side="left")
tokenizer.pad_token = tokenizer.eos_token

f = open("songsGpt2.txt", "a", encoding="utf-8")
i = 0

for prompt in prompts:
    output = ""
    i += 1
    while len(output) < 1000:
        model_inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
        generated_ids = model.generate(
            **model_inputs, 
            max_new_tokens=1024 - len(prompt), 
            do_sample=True, 
            repetition_penalty=1.01
        )
        output = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        print(f"Song generated length of {len(output)}")
    
    print(f"Song {i} finished\n")
    f.write("\n*****SONG STARTS HERE*****\n")
    f.write(output)
f.close()


