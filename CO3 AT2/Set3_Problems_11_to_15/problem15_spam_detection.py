P_spam=0.25
P_not=0.75
P_offer_spam=0.80
P_offer_not=0.10

P_offer=P_offer_spam*P_spam + P_offer_not*P_not
P_spam_offer=(P_offer_spam*P_spam)/P_offer

print("Probability Spam | Offer =",round(P_spam_offer,4))
print("Percentage =",round(P_spam_offer*100,2),"%")
