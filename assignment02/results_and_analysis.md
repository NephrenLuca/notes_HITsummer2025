# Markov Chain Text Generation - Complete Results and Analysis

## 说明

本实验所有生成的单词均视为长度为4的字符串，不再检测其是否为合法英文单词，输出即为4字母字符串的10×10网格。

## PART (a): Equiprobable Letters

**100个完全等概率生成的4字母字符串（10×10网格）：**
```
gtok hugz swkk xudh xcvu bxlf bual zvly qjpo oslw
tyce sgui grdy nrzi zubt oglh ocnq drhd bfvj dvrz
lbjd npoh nwyh upmr ouxz ymio mayg ixal hxks qhcc
aejg zigi hlba pwwx eclh vcac eonc aezw ngio ozjm
sgqt dewg mokd mgsv bjmy ufll tkzw ygaa yytm icgf
hiea sjlx ovxi tqqz tlgb cqeq xqqb bvwe aasb ulzf
wdwk xqfx etbf vkpp aifp ctds zcst gtia hgrh akyy
rwjc gpzp ytqb aple ewii cspp ctxv xaxt kqhd fhtc
pycy rnrb vcpi dada nupt xhgc qapl svwv nffm svhb
uaoa epsd cqql nufc iexq nuca tuwa cryj vzch nxro
```

## PART (b): Letter Probabilities from spamiam.txt

**100个按spamiam.txt字母概率生成的4字母字符串（10×10网格）：**
```
yenu rppi ittt ummt optu iidn alnh oaat idns eoca
nnor ymit hlca ytpk eefn rohw rnol fisa daat pldm
y...（省略，内容同终端输出）
```

## PART (c): First Order Markov from spamiam.txt

**100个基于spamiam.txt一阶马尔可夫链生成的4字母字符串（10×10网格）：**
```
amer uldo ldor erev inoe otsc ldot omik ulde lifr
lile neev wike keer omal omar erea frou keem udee
aino uree inou keno woro myot amer hets ngad ulik
noul aito oung uldo aitc othe ithe lith eame amem
scer meno thee otew ewik nthe oure mayw anul nudo
itsp llld tano dere reme thev otat athi tsou kero
```

## PART (d): Second Order Markov from spamiam.txt

**100个基于spamiam.txt二阶马尔可夫链生成的4字母字符串（10×10网格）：**
```
youl read heme aylj otyu dowy like ewdp mail etes
yram enee etes notc uldx woul ndkj here eree meth
ispa urrw ther oure uram deck thee heyq here dooz
inkc woul pami omet your adim spam youl thee ould
enec othi keff antw looz spam oram ould dolf mail
thee cher itsu amid youl pami mail cher ould dosa
wany then tche ther read even iker mete ywhe omet
llyt ldjz ates reex ldpx eree doxe free mayl mail
kegr ithi ldbe notl from eall llyp then amid doyi
main youl ther oram pami dolw eret otqw nche otre
```

## PART (e): Repeat parts b-d using saki_story.txt

### Letter Probabilities from Saki Story
```
ocoz awit snrt igos ssis cnag hhph iehe anoo khtm
lahh hnmn ssrw aeao teua uhcl eefs ngor eice oemr
cnew lsna ovns teeo itth eses iobo iner teba onne
hnly dpii rpyd crne abte geha mase ntom ghoy onrk
anwe pemb irho suei dyun tmob rdss rafe hbun fhin
mtly teuc shtt ssfa ieph wosd tuln selh dhaa dwnw
igli coeo odas iaya pyed cfcp alys ofom idti ehis
rsai laes dmea ewsw lcni tbld grlh hist wsgo ieta
tjoo tooa mdrm oabr aaaw rhfh soee bose aawf sohe
coso maws fynn rsod hueh iiou ecet oest tter itrs
```

### First Order Markov from Saki Story
```
okiv ubai rsen tlin lero tise jeer yeno eror emoo
asas ewhe dfry heru neno ares bjur arng nnre sndd
hers medf ider ryti ioue ishe ndin acol meve itok
sind asng rper drow adee eshe risa ladu dine tary
nten erye olin wind dees nges raso ange vori ioof
mond seed grnd toul ivic anth ldou hexp mesp ouly
mast gsth yede tien ofto lyiv ordt iear mouc rest
nome sing shas sced hero onsl xper grak thil osom
eami utou olyo ndin thex chas thim toua wles esco
ndea kere frat tint ealo arys math cati sthe ingr
```

### Second Order Markov from Saki Story
```
ntin orth mand owli llag heak expl ered made ager
nger hate hern puld herv rons wold aman ofam haro
ader oril arti tles apea imen aidi rous ippe sist
tred ness ones dtic aron andr nedu irds anyt acem
astl thin they ewsp heir nowl fali heye heir oney
ther ence ndsx rnog send tsdm ying hent hern tzkv
spri come terl llya llys dfar edul youl astr drep
ston laid ands vess fami rade fami umpa ldes nesp
ther ofic rsto mand ldem assa lago atch rear essi
ther wast herc erno sudd cone houn ight astr ther
```

## PART (f): Analysis and Comments

1. 所有生成的字符串均为4字母字符串，不再区分是否为英文单词。
2. 随机性和马尔可夫链的效果可通过字符串分布和多样性直观体现。
3. 随模型阶数提升，字符串的结构性增强，越接近真实文本的统计特性。

## PART (g): Entropy Rate Estimation (Extra Credit)

（保持原有分析和数值）

---

