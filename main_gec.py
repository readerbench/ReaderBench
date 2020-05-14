from rb.processings.gec.gec_correct import GecCorrector

if __name__ == "__main__":
    
    corrector = GecCorrector(d_model=768)
    decoded_senntence = corrector.correct_sentence('Am mers la magazi sa cuumper leegume si fructe de mancaat.')
    print(decoded_senntence)