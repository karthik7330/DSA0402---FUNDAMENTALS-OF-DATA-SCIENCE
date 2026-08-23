branchA=(4.1,4.5)
branchB=(3.8,4.2)
print("Branch A:",branchA)
print("Branch B:",branchB)
if branchA[0]>branchB[1]:
    print("Branch A performs significantly better.")
elif branchB[0]>branchA[1]:
    print("Branch B performs significantly better.")
else:
    print("Confidence intervals overlap. No significant difference.")
