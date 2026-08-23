P_disease=0.02
P_no=0.98
P_pos_given_dis=0.95
P_pos_given_no=0.10
P_pos=P_pos_given_dis*P_disease + P_pos_given_no*P_no
posterior=(P_pos_given_dis*P_disease)/P_pos
print("Posterior Probability =",round(posterior,4))
print("Percentage =",round(posterior*100,2),"%")
