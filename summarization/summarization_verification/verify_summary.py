football_prompt_text = """ 
Summarize the text following  '<BEGIN DOCUMENT TEXT>'  Focusing only on the following aspects:

Aspects:
1) Legacy and Recognition: if no info for an aspect -> do not output anything 
2) Influence on Spanish Football Tactics and Style: if no info for an aspect -> do not output anything
3) Death 

Format: For each aspect a bullet, according to the above aspects

<BEGIN DOCUMENT TEXT>    
==Club career==
Born in [[A Coruña]], he was transferred to [[Madrid]] as a young man, and began practicing football at the Nuestra Señora del Pilar school, which once was one of the city's football cradles, and from there he entered the training categories of [[Real Madrid CF|Real Madrid]] at the end of the 1918-19 season.<ref name=vida>{{cite web |url=http://hemeroteca.abc.es/nav/Navigate.exe/hemeroteca/madrid/abc/1950/11/15/019.html |title=El famoso ex-jugador de fútbol Monjardín muere víctima de un accidente de automóvil |trans-title=The famous ex-football player Monjardín dies the victim of a car accident |language=es |publisher=Diario ABC|accessdate=10 June 2022}}</ref> In that same season, and despite his early age of just 15, he debuted with the first team in a [[Campeonato Regional Centro|Central Regional Championship]] match against [[Racing de Madrid]], and he quickly became one of the club's benchmarks at the time. He soon evolved from his position of midfielder to forward, which he no longer gave up until the end of his career.<ref name=monjardin>{{cite web|url=https://www.marca.com/blogs/ni-mas-ni-menos/2015/03/21/juanito-monjardin-el-de-las-piernas.html |title=Juanito Monjardín, el de las piernas torcidas |trans-title=Juanito Monjardín, the one with crooked legs |language=es |publisher=Diario Marca|author=Jesús Ramos |accessdate=10 June 2021}}</ref>

Between the aforementioned Regional Championship and the Copa del Rey, he accumulated a total of 55 goals in 74 games. At the time of his retirement in 1929 (aged 26), he was the second top scorer of the Madrid team, only surpassed by the 68 goals from teammate [[Santiago Bernabéu Yeste|Santiago Bernabéu]].{{Citation needed|date=June 2022}} One of the reasons for his early retirement was the arrival of two players who ended up also being attacking and historical references of the club, the Valencian [[Gaspar Rubio]] and [[Jaime Lazcano]] from [[Navarra]], both younger than him, and both ended up breaking his goalscoring record at the club. The same season of his retirement La Liga was inaugurated, and by playing only one game, which was also the only one he played that season, he became one of the 19 club players to appear in that historic first edition.<ref>{{cite web|url=http://www.bdfutbol.com/es/t/t1928-292.html|title=Plantilla Real Madrid 1928-29|publisher=Portal digital BDFutbol|accessdate=10 June 2022}}</ref>

In 1943, years after his professional retirement, the white club organized a tribute match in his honour between the people of Madrid and [[FC Barcelona|Barcelona]], ending with a one-goal tie.<ref>{{cite web|url=http://hemeroteca.mundodeportivo.com/preview/1943/10/23/pagina-2/651251/pdf.html|title=''Real Madrid, 1 - Barcelona, 1''(PDF)|publisher= Diario El Mundo Deportivo|accessdate=10 June 2022}}</ref>

==International career==
Being an [[Real Madrid CF|Madrid FC]] player, he was eligible to play for the [[Madrid autonomous football team|'Centro' (Madrid area) representative team]]], and he was part of the squad that participated in two tournaments of the [[Prince of Asturias Cup]], an inter-regional competition, in [[1922–23 Prince of Asturias Cup|1922–23]] and [[1923–24 Prince of Asturias Cup|1923–24]], and although the first campaign ended with a shocking quarter-final exit at the hands of [[Galicia national football team|Galicia]], in which Monjardin scored Madrid's consolation goal in a 1–4 loss, the second campaign was much better, largely thanks to Monjardin as he scored twice in their 2–1 win over a [[Andalusia national football team|Andalusia XI]] in the semi-finals, followed by what appeared to have been an [[Overtime (sports)#Association football|extra-time]] winner against [[Catalonia national football team|Catalonia]] in the [[1924 Prince of Asturias Cup Final|final]] to seal Madrid's second Prince of Asturias Cup title, but a last-minute equaliser from [[Emili Sagi-Barba]] forced a replay in which he scored again, netting twice in the first-half, but his efforts were in vain as Catalonia took the title with a 3–2 win.<ref name=Prince>{{cite web |url=http://www.cihefe.es/cuadernosdefutbol/2009/09/la-copa-principe-de-asturias/ |title=La Copa Príncipe de Asturias |trans-title=The Prince of Asturias Cup |language=es |publisher=[[:es:Centro de Investigaciones de Historia y Estadística del Fútbol Español|CIHEFE]] |author=Vicente Martínez Calatrava |date=17 August 2009 |accessdate=5 June 2022}}</ref> The silver lining being that with five goals, he was the top goal scorer of the 1923–24 Prince of Asturias Cup, and with a total of six goals in the competition, he is the [[Prince of Asturias Cup#All-time top goalscorers|joint all-time top goalscorer]] of the Prince of Asturias Cup along with [[José Luis Zabala]] and [[Juan Armet|Kinké]].

He made his debut for the [[Spain national football team|Spain national team]] in [[Lisbon]] on 17 December 1922 against [[Portugal national football team|Portugal]], scoring the winning goal of a 2-1 win in the 82nd minute. In his next cap against [[France national football team|France]] on 28 January 1923, he scored a brace in a 3-0 win, and coincidentally, the author of the third goal was Zabala. In total, he was capped four times, scoring three goals.<ref name=EU>{{cite web|url=https://eu-football.info/_player.php?id=14272 |title=Juan Monjardín |work=EU-football.info |access-date=10 June 2022}}</ref>

<END DOCUMENT TEXT>

Format instructions: a numbered bullet for each aspect:
1) Legacy and Recognition:  
2) Influence on Spanish Football Tactics and Style: 
3) Death: 

"""
 
data = {'text': [football_prompt_text]}
 
verify_summary_prompt = """
Prompt:
{text}

Summary:
{summary}

Task: You act as an LLM Judge:

Does the Summary expresses important and accurate facts from the prompt ?
For each sentence or fact stated in the Summary, throughly check if it was correctly taken from the Text to Summarize.
"""
 
incorrect_summary = """
1) Legacy and Recognition:
- Organized a tribute match in his honor between Madrid and Barcelona in 1943.
- Joint all-time top goalscorer of the Prince of Asturias Cup with five goals in 1923-24.

2) Influence on Spanish Football Tactics and Style:
- Evolved from midfielder to backward, which was a significant position change in football tactics.

3) Death:
- Died in a car accident in 1950.
"""
   
def verify_summary(cfg):
    # Create llm
    cfg.summary.verify.create_llm()
    # Summarize 
    df = pd.DataFrame(data)
    cfg.summary.verify.generate_batch(prompt='{text}', df=df, resp_col = 'summary')
    print('Summary\n----------------\n')
    print(df.summary.iloc[0])
    print('Verify Summary. Expected - Accurate (if teh llm in prev step did summarize well)\n----------------------------------------------\n')
    cfg.summary.verify.generate_batch(prompt=verify_summary_prompt, df=df, resp_col = 'resp_verify_summary')
    print(df.resp_verify_summary.iloc[0])
    print('Verify Incorrect Summary (backward instead of forward). Expected - Detect mistake\n-------------------------------------\n')
    df.iloc[0, df.columns.get_loc('summary')] = incorrect_summary
    cfg.summary.verify.generate_batch(prompt=verify_summary_prompt, df=df, resp_col = 'resp_verify_summary')
    print(df.resp_verify_summary.iloc[0])