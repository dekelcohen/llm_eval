from pathlib import Path
import pandas as pd
from vllm import LLM, SamplingParams

# Read texts to classify
label_col_name = 'openchat_houthis_sentiment'
dataset_path = 'data/2024-01-12_15-08-46__2024-01-21_18-42-15_HouthisTweet_classified'

df = pd.read_excel(dataset_path+ '.xlsx')


houthis_classification_desc = '''
In recent weeks, Yemeni Houthi militants, which are backed by the iranian regime, have repeatedly targeted and attacked U.S. personnel, military bases, and commercial vessels in the Red Sea, resulting in significant disruptions to maritime traffic and raising concerns about regional stability and security. As a response, the U.S., in collaboration with several other countries, has initiated Operation Prosperity Guardian to collectively address security challenges in the southern Red Sea and the Gulf of Aden, with the primary goal of ensuring the freedom of navigation in the area. On January 12, 2024, U.S. and U.K. forces conducted multiple airstrikes against Houthi targets as part of this operation.

According to Lt. Gen. Douglas Sims, the director of the US military's Joint Staff, the initial strikes on January 12 successfully achieved their objective of damaging the Houthis' ability to launch complex drone and missile attacks similar to the ones they conducted earlier. However, despite causing significant damage to Houthi capabilities, subsequent assessments indicated that the strikes did not entirely deter further attacks on shipping in the region. Yemeni sources reported casualties among Houthi fighters, including members of Hezbollah, the Iranian Revolutionary Guards, and Iraqi militants.

Reactions to these military actions have been varied. Domestically, the Yemeni government strongly condemned the Houthi attacks and affirmed its right to enhance security in the Red Sea region. On the other hand, Houthi officials labeled the airstrikes as blatant aggression and vowed retaliation. Massive protests erupted in Houthi-controlled cities denouncing the U.S. and British military actions.

Internationally, responses ranged from expressions of support to condemnation. While some countries, including the U.S. and the U.K., justified the strikes as necessary for safeguarding maritime navigation and addressing security threats, others, such as Iran and Syria, condemned them as violations of Yemen's sovereignty and international law. Notably, China urged restraint and highlighted the need to avoid further escalation in the region.

Given these developments and the diverse range of reactions, your task is to classify tweets related to American strikes against the Houthis into one of the following categories: Class 1, Class 2, Class 3, Class 4, Class 5, or Class 0. Below, we provide clear and detailed guidance for each class based on the spectrum of opinions and reactions observed.

Class 1: Supportive of U.S. actions against the Houthis.

Tweets that endorse U.S. and coalition operations against the Houthis, emphasizing the need for repercussions for Houthi aggression.
Statements welcoming military actions as necessary and overdue, highlighting the Biden administration's warnings to the Houthis.
Expressions of support for attacking the Houthis in response to their actions, linking it to national security concerns.
Pay attention, tweets with minor criticism of the Biden administration, as long as they primarily endorse the military actions against the Houthis, should still be classified under Class 1.
If the tweet explicitly endorses targeting Tehran as a response to Iranian support for militant groups or aggression against U.S. interests or allies, it should be classified as Class 1.
Class 2: Expressing a nuanced view on the American attacks against the Houthis.

Tweets presenting mixed opinions on U.S. attacks in Yemen on the Houthis, acknowledging both positive and negative aspects without firmly aligning with either support or objection.
Pay attention, if the tweet primarily expresses support for the strike and emphasizes the importance of taking action against the Houthi aggression, while also containing elements of critique towards the Biden administration's handling of the situation in a manner that does not include constitutional concerns, regional stability concerns, or isolationist views, should be classified under Class 1, as long as the overall tone of the tweet leans more towards endorsing the military actions.
Class 3: Objecting to military operations against the Houthis due to potential escalation of violence.

Tweets expressing concerns about the consequences of military operations against the Houthis, including escalation of violence and destabilization of the region.
Statements criticizing airstrikes against the Houthis as violating international law or sovereignty, with a focus on promoting peace and stability.
Tweets expressing concerns about the effectiveness of military actions in Yemen against the Houthis and advocating for alternative, non-violent approaches to conflict resolution.
Class 4: Opposing the attacks against the Houthis because of opposing any U.S. involvement in foreign conflicts, reflecting isolationist views.

Tweets condemning U.S. attacks against the Houthis because of objection to U.S. involvement in foreign wars and advocating for prioritizing domestic affairs over military engagements.
You will be classified in this category only if the tweet talks about it in the context of the Houthis or the attacks in Yemen. If it is talking about other places, do not classify as this category.
Class 5: Opposing the American attacks on the Houthis, citing constitutional concerns.

Tweets explicitly condemning U.S. airstrikes against the Houthis as unconstitutional due to lack of congressional approval.
Statements asserting that only Congress has the authority to declare war and criticizing the Biden administration for bypassing this process in the Houthi subject.
Expressions of reluctance to support broader military involvement in Yemen without clear congressional authorization and debate.
If the tweet contains a negative stance on the attacks in Yemen and can be classified under more than one of the negative classes (3, 4, 5), classify the tweet by analyzing under which of the classes most of the tweet falls under.

Class 0: Unopinionated tweets or tweets that are not related to the american houthi conflict
Tweets that do not express any opinion on the houthi american situation and the american attacks on the houthi rebeles.
Include tweets that are not related to the houthies and their actiones.
expressions that are related to different conflicts apart from the houthies will also be included in class 0.
Pay attention, tweets that mention the need to release all hostages and the war in Gaza without mentioning Houthi attacks in the Red Sea should be classified under class 0.
Pay attention, if the tweet contains the words "Houthis," "Red Sea," "Iranian proxies," and such, it should be classified under class 0 if it does not express an opinion regarding the Houthi attacks or the American attacks. If a tweet contains those words without expressing an opinion, it should be classified under class 0.



Examples:
Tweet: "This action by U.S. and British forces is long overdue, and we must hope these
		operations indicate a true shift in the Biden Administration’s approach to Iran and its
		proxies that are engaging in such evil and wreaking such havoc. They must
		understand there is a serious price to pay for their global acts of terror and their
		attacks on U.S. personnel and commercial vessels. America must always project
		strength, especially in these dangerous times."
ANSWER: 1

Tweet: "I welcome the U.S. and coalition operations against the Iran-backed Houthi terrorists
		responsible for violently disrupting international commerce in the Red Sea and
		attacking American vessels. President Biden’s decision to use military force against
		these Iranian proxies is overdue."
ANSWER: 1

Tweet: "The United States and our allies must leave no room to doubt that the days of unanswered terrorist 
		aggression are over."
ANSWER: 1

Tweet: "The United States carries a special, historic obligation to help protect and defend
		these arteries of global trade and commerce. And this action falls directly in line with
		that tradition. That is clearly reflected in both our national security strategy and the
		national defense strategy. It is a key conviction of the president and it is a
		commitment that we are prepared to uphold."
ANSWER: 1

Tweet: "The President’s strikes in Yemen are unconstitutional. For over a month, he
		consulted an international coalition to plan them, but never came to Congress to
		seek authorization as required by Article I of the Constitution. We need to listen to
		our Gulf allies, pursue de-escalation, and avoid getting into another Middle East war."
ANSWER: 5

Tweet: "The United States cannot risk getting entangled into another decades-long conflict
		without Congressional authorization. The White House must work with Congress
		before continuing these airstrikes in Yemen."
ANSWER: 5

Tweet: "We will see if these strikes deter Iran and its proxies from further attacks; I have my
		doubts. History teaches that only devastating retaliation will deter Iran, as when
		President Trump killed their terrorist mastermind in 2020 and President Reagan sank
		half their navy in 1988. That bold, decisive action is the opposite of what we’ve seen
		from Joe Biden for three years."
ANSWER: 2

Tweet: "In this particular instance, I do support military reprisals and military attacks to deter
		attacking our ships, but [the Biden administration] shouldn’t be allowed to do that
		without permission."
ANSWER: 5

Tweet: "the strikes showed a &quot;complete disregard for international law&quot; and were
		&quot;escalating the situation in the region."
ANSWER: 3

Tweet: "The U.S. air strikes on Yemen are another example of the Anglo-Saxons&#39; perversion
		of UN Security Council resolutions."
ANSWER: 3

Tweet: "These attacks are a clear violation of Yemen&#39;s sovereignty and territorial integrity,
		and a breach of international laws. These attacks will only contribute to insecurity
		and instability in the region."
ANSWER: 3

Tweet: "I Called for restraint and &quot;avoiding escalation&quot; after the strikes and said it was
		monitoring the situation with &quot;great concern&quot;."
ANSWER: 3

Tweet: "I call on all parties involved not to escalate even more the situation in the interest of
		peace and stability in the Red Sea and the wider region."
ANSWER: 3

Tweet: "U.S. Strikes Won’t Achieve What a Gaza Cease-Fire Could in the Middle East"
ANSWER: 3

Tweet: "This is why I called for a ceasefire early. This is why I voted against war in Iraq,  and why I am against the attacks in Yemen today.
		Violence only begets more violence."
ANSWER: 3

Tweet: "We need a ceasefire now to prevent deadly, costly, catastrophic escalation of
		violence in Yemen.
ANSWER: 3

Tweet: "I would not support us being pulled into a broader war because of the operation in Yemen."
ANSWER: 3

Tweet: "As representatives of the American people, Congress must engage in robust debate
		before American servicemembers are put in harm’s way in Yemen and before more U.S.
	   taxpayer dollars are spent on yet another war in the Middle East."
ANSWER: 4

Tweet: "Only Congress has the power to declare war. I have to give credit to [Rep. Ro
		Khanna] here for sticking to his principles, as very few are willing to make this
		statement while their party is in the White House."
ANSWER: 5

Tweet: "The President must come to Congress for permission before going to war.
		Biden can not solely decide to bomb Yemen. 
		And what is the condition of Secretary of Defense Lloyd Austin? Is he still laid up in
		the hospital?
		Biden [administration] wants to fund war in Ukraine, control the war in Israel, arm
		Taiwan and prep for war with China, and is now going to war in the Middle East???
		All with a wide open border, millions invading, and millions of got aways?! 
		This is insanely out of control!"
ANSWER: 5

Tweet: "his is where we should put party aside and stand for the oath we all took: Congress
		alone decides if we go to war. I join my colleagues on both sides insisting we follow
		the Constitution."
ANSWER: 5

Tweet: "POTUS is violating Article I of the Constitution by carrying out airstrikes in Yemen
		without congressional approval. The American people are tired of endless war."
ANSWER: 5

Tweet: "The President must come to Congress for permission before going to war. Biden can not
		solely decide to bomb Yemen."
ANSWER: 5

Tweet: "The Houthi attacks are a result of Biden's weakness. This country needs a change. Trump2024."
ANSWER: 0

Tweet: Ramadan Mubarak to all observing this sacred month.

This Ramadan is particularly difficult, w millions of Palestinians on the brink of starvation and under siege. We must secure a ceasefire to save countless lives.

May the blessings of this month bring a just and lasting peace".
	 ANSWER: 0

Tweet: 
"Biden administration intelligence officials testified that Iran has “orchestrated” attacks against America and Israel since October 7. 

The ayatollahs continue to order attacks because they know Joe Biden won’t respond. https://t.co/sbtGLvmplb"

-"Since taking office, Biden has enriched the Iranian regime to the tune of $100 BILLION dollars.

Stop appeasing terrorists, @POTUS."	
ANSWER: 0

Tweet: 
-I urge @POTUS and partners in the region to continue with negotiations to reach a mutual ceasefire agreement that is in the best interests of Israel, Palestinians, and stability and peace in the region. The current situation in the Middle East is not sustainable.
ANSWER: 0

Tweet: 
- Our nation is facing over $34 trillion of debt, and our southern border is wide open. Yet the D.C. Cartel continues to prioritize foreign nations while putting America last. We must secure America’s borders first!

Mayorkas was the first ever Homeland Security Secretary to be impeached by the House."
ANSWER: 0


Tweet: 
-"Without a vote from Congress, the Biden Administration is sending another $300 million to fund the war in Ukraine.

This is on top of the $113 BILLION in funding American taxpayers have already sent over there. 

America should be worried about securing our OWN border, not Ukraine’s. 
https://t.co/x1PzCoVjd8"
ANSWER: 0
'''


prefix = houthis_classification_desc # Note the Tweet: {input} place holder was deleted and instead, formatted below
prompts = df.post_text.apply(lambda text: f'Tweet: {text}\nAnswer:' ).tolist()

# Sample prompts.
"""
prompts = [
    "Hello, my name is",
    "The president of the United States is",
    "The capital of France is",
    "The future of AI is",
]
"""
# Create a sampling params object.
sampling_params = SamplingParams(temperature=0.0)

# Create an LLM.
llm = LLM(model="openchat/openchat-3.5-1210")

generating_prompts = [prefix + prompt for prompt in prompts]

print("-" * 80)

# The llm.generate call will batch all prompts and send the batch at once
# if resources allow. The prefix will only be cached after the first batch
# is processed, so we need to call generate once to calculate the prefix
# and cache it.
outputs = llm.generate(generating_prompts[0], sampling_params)

# Subsequent batches can leverage the cached prefix
outputs = llm.generate(generating_prompts, sampling_params)

# Print the outputs. You should see the same outputs as before
gen_texts = []
for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    gen_texts.append(generated_text)
    print(f"Generated text: {generated_text!r}")
    #print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
    
df['gen_resp'] = gen_texts   

def parse_response(resp: str):
    import re
    lst_str_nums = re.findall(r'(?:ANSWER:\s*)?(\b\d+\b)', resp)
    numbers = [int(str_num) for str_num in lst_str_nums]
    return numbers
    
def get_class(resp):
    numbers = parse_response(resp)
    num = numbers[0] if len(numbers) > 0 else -1
    return num
        
df[label_col_name] = df.gen_resp.apply(get_class)
                    
df.to_parquet(dataset_path + '.parquet')


    