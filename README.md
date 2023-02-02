# TEG

### dataset 다운로드  
[다운로드](https://kaistackr-my.sharepoint.com/:f:/g/personal/swkim_kaist_ac_kr/EuCJ-EDjyCJGtFyQP3sBKVEBgRIatTws5aaCgXZdgq1dLA?e=RQVG4s)  
data 폴더 그대로 최상위 경로에 넣어주면 됨  
### Result 파일  
[Result Summary.xlsx](https://kaistackr-my.sharepoint.com/:x:/r/personal/swkim_kaist_ac_kr/_layouts/15/Doc.aspx?sourcedoc=%7B5667ACAD-4E6B-471E-94FB-255277C0FE09%7D&file=Result%20Summary.xlsx&action=default&mobileredirect=true)



---

### result  
Seed 5번 실험이라면 seed 하나하나에 대한 결과는 `result.txt` 파일에,  
5개 Seed 전체에 summary 결과는 `final_result.txt` 파일에 저장됨.  

---

### argument  
#### ours  
`lr` : learning rate  
`epochs` : 2000개로 default => early stopping 됨  
`seed` : 시작 seed (default : 1)  
`num_seed` : 반복할 seed 개수 (seed가 1이고, num_seed가 5라면, 1~5번 시드 돌아감)  
`patience` : early_stopping patience (default : 10)  
`summary` : result, final_result에 표시할 실험이름 (default : timestamp)  
`way` / `shot` / `qry` : way, shot, qry  
`episodes` : train할 때 1 epoch당 episode 개수  
`meta_val_num` : valid할 때 1 epochs 당 episode 개수  
`meta_test_num` : test할 때 1 epochs 당 episode 개수 (30개의 accuracy)  
`l1` : l1 loss lambda  
`l2` : l2 loss lambda (GCN loss) 
`local_proto` : local proto 반영 비율 (default 0.5)  
`global_proto` : global proto 반영 비율 (default 0.5)  
`final_result` : final result 저장 경로  
`n_layers` : EGNN hidden layer 개수 (늘리면 NAN의 저주가 일어날 수도 있음)  

#### TENT  
`tent_lr` : tent learning rate  
`tent_hidden` / `tent_hidden2` : tent hidden dim  
'tent_dropout` : tent dropout  

---  
### Reported Experiments Settings  
`TENT` : Train; 500 epochs (=500 episodes) / Test; train 50 epochs마다 Valid 50epi & Test 50epi해서 Best Valid일 때 Test Accuracy로 report / Qry = 10 / 5번 repeat  
`GPN` : Train the model over 300 episodes and early-stopping strategy / 50 meta-test tasks / 50 meta-test tasks / Qry = # of support size (K개) / 10번 repeat
# TEG-KDD
