from rb.processings.sentiment.SentimentAnalysis import SentimentAnalysis
from rb.core.lang import Lang

if __name__ == "__main__":
    sa = SentimentAnalysis(Lang.RO, model_type="base")
    
    rev0 = "un produs de foarte bună calitate. foarte mulțumit de achiziție"
    rev1 = "dezamăgit de acest produs. raport calitate preț foarte slab"
    rev2 = "produs ok"
    rev3 = "s-a stricat după 2 zile. nu cumpărați!"
    rev4 = "foarte mulțumit. recomand"
    rev5 = "ok"
    rev6 = "Costul este exagerat pentru asemenea performante slabe. Nu recomand. asemenea performante slabe. Nu recomand. asemenea performante slabe."
 
    scores = sa.process_text([rev0, rev1, rev2, rev3, rev4, rev5, rev6])
    print(scores)
