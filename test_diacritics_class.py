from rb.processings.diacritics.DiacriticsRestoration import DiacriticsRestoration


if __name__ == "__main__":
    
    dr = DiacriticsRestoration(model_name="small")
    
    # s = "Marcel Ciolacu a fost externat din Spitalul Militar unde a fost pentru investigații după ce i s-a făcut rău în timpul conferinței de presă de la sediul PSD. Medicii au anunțat că acesta a avut o cădere de calciu. De asemenea, i-au făcut un test rapid pentru coronavirus, rezultatul fiind negativ."
    s = "Sal?ut ce!mai!faci sî m!<->ai zici ROmanIă. Asemenea si tîe, pana data v<>i.,Î,.,.,toare!"
    s0 = dr.process_string(s, mode="replace_all")
    s1 = dr.process_string(s, mode="replace_missing")

    print(s)
    print(s0)
    print(s1)
