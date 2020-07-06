from gensim.summarization import summarize

from rb.summarization.systems.summarizer_abc import *

import logging
logging.getLogger("gensim").setLevel(logging.WARNING)


class Gensim(Summarizer):

    def __init__(self):
        Summarizer.__init__(self)

    def summarize(self, doc, lang=Lang.EN, parser=None, ratio=0.2, word_count=None) -> Iterable[str]:
        if not parser:
            parser = self.parser

        doc_sentences = parser.tokenize_sentences(doc)
        return summarize('\n'.join(doc_sentences), ratio=ratio, word_count=word_count, split=True)


def main():
    gensim_summarizer = Gensim()

    doc = """
            Elephants are large mammals of the family Elephantidae
            and the order Proboscidea. Two species are traditionally recognised,
            the African elephant and the Asian elephant. Elephants are scattered
            throughout sub-Saharan Africa, South Asia, and Southeast Asia. Male
            African elephants are the largest extant terrestrial animals. All
            elephants have a long trunk used for many purposes,
            particularly breathing, lifting water and grasping objects. Their
            incisors grow into tusks, which can serve as weapons and as tools
            for moving objects and digging. Elephants' large ear flaps help
            to control their body temperature. Their pillar-like legs can
            carry their great weight. African elephants have larger ears
            and concave backs while Asian elephants have smaller ears
            and convex or level backs.  
        """

    big_doc = """
                Nikolai Ryzhkov, the former Soviet prime minister, was reported running a
                distant second, averaging about 20 percent of the votes cast but doing better
                in rural areas where the Communist Party had mobilized support among
                farmworkers.;    Although a strong vote for Ryzhkov and perhaps other
                conservative candidates in Russia's smaller cities and rural areas could
                diminish Yeltsin's margin, he appeared certain of the absolute majority of
                more than 50 percent he needed to win the presidency on the first ballot,
                avoiding a runoff election.;    The likely size of that victory, moreover,
                should not only assure Yeltsin the new executive presidency of the Russian
                federation, the largest Soviet republic, but also give him the mandate he
                sought for sweeping political and economic changes in the Soviet Union as a
                whole.;    A beaming Yeltsin was greeted by chants of "Victory! Victory!" from
                scores of supporters as he voted in Moscow. After dropping his ballot in a
                box, he raised his fist and clasped his hands above his head in triumph.;   
                "This is a celebration of the Russian federation's sovereignty," said Yeltsin,
                noting that Wednesday was a holiday in the republic to mark the first
                anniversary of Russia's move to wrest control of its resources away from the
                Kremlin.;    Soviet President Mikhail Gorbachev refused to say who got his
                vote after he cast his ballot in Moscow. But he stressed that he is willing to
                work with the winner.;    "I am ready to cooperate with anyone who will be
                elected by Russians," Gorbachev said. "There will be no problems from my
                side.";    It is clear, however, that a Yeltsin victory would represent a
                distinct threat to Gorbachev, who has never faced a popular election. Yeltsin
                would be able to claim that he alone has a democratic mandate to speak for the
                people of the Soviet Union's largest republic.;    Ryzhkov, at another polling
                station, said: "Social tension is high, and if we make drastic decisions that
                affect people's lives, there will be unpredictable consequences. . . . I
                cannot allow Russia and the rest of the country to be divided into two camps,
                one camp for Gorbachev and the other for Ryzhkov or some other person.";    In
                other results, the radical leaders of Moscow and Leningrad, Gavriil Popov and
                Anatoly A. Sobchak, appeared to have won election over conservative rivals as
                executive mayors with greater executive powers, according to the independent
                Russian Information Agency.;    A proposal to change the name of Leningrad
                back to St. Petersburg reportedly won the approval of 55 percent of the voters
                in a non-binding referendum, but the ballot counting was not yet complete.;   
                Even from the preliminary reports gathered by the Russian Information Agency
                and other independent news organizations from across Russia, the size of
                Yeltsin's victory was impressive.;    In constituencies as diverse as the
                Soviet navy's Pacific fleet, fishermen and merchant sailors at sea, mining
                towns beyond the Arctic Circle, Siberia's great industrial centers, the
                ancient cities of European Russia and regions along the Chinese and Mongolian
                borders, Yeltsin won the support of 55 percent to 80 percent of the voters,
                according to unofficial reports from the Russian Information Agency.;    Such
                a victory, demonstrating a strong majority support across the country, should
                boost Yeltsin's political stature considerably.; Boris Yeltsin ahead in
                election; (box) Born: Feb 1, 1931, in Butka, Siberia, in Russian republic;
                (box) 1955: Construction worker in Sverdlovsk; (box) 1976: First secretary,
                Sverdiocsk District Central Committee; (box) 1985: First secretary of Moscow
                Communist Party.; (box) 1987: Outburst against conservative archrival Yegor
                Ligachev leads to Yeltsin's ouster from Politburo.; (box) 1989: Bounces back
                from disgrace; wins 89% of vote to be Moscow's representative in New Congress
                of People's Deputies, the national parliament.; (box) 1990: Republic's
                parliament elects him president of Russia.; (box) June 12, 1991: In Russia's
                first popular election, Yeltsin leads by large margin in race for president.
            """

    # doc_as_list = gensim_summarizer.parser.tokenize_sentences(doc)
    # doc_as_list = map(lambda s: gensim_summarizer.parser.preprocess(s, Lang.EN.value), doc_as_list)
    #
    # summary = gensim_summarizer.summarize('\n'.join(doc_as_list))
    # for sent in summary:
    #     print(sent)

    # summarize_dataset(dataset_type=DatasetType.DUC_2002, summarizer=gensim_summarizer)


if __name__ == "__main__":

    main()
