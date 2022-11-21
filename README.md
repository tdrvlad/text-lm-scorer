# text-lm-scorer

## TODO
Function that outputs a Casual Language Model output probability for each word (token) in a given sentence. Bonus: for each word (token) return also the word the model considers the most probable. Example:
```
lm_scorer('Today is a nice day. Where shall we go?')
```
returns:
```
 is (0.017149414867162704) - ,
 a (0.1455073356628418) -  the
 nice (0.002566296374425292) -  time
 day (0.4161650836467743) -  day
. (0.1210041344165802) -  for
 Where (0.0010273606749251485) - 

 shall (0.0016869400860741735) - ver
 we (0.4552111327648163) -  we
 go (0.38437604904174805) -  go
? (0.16317293047904968) - ?
```
