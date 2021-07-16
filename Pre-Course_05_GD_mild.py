import sympy as sym
from sympy.abc import x
#미분을 파이썬으로 가능하다.
#기호적으로 이해하는 sympy를 사용하자.
#증가시키고 싶다면 미분값을 더하고 감소시키고 싶으면 미분값을 뺀다. 
#이를 이용해 최적화에 이용한다. 더하며 최대화할때: 경사상승법이라 부른다.
#미분값을 빼주며 목적함수를 최소화하는게 경사하강법.

a=sym.diff(sym.poly(x**2 + 2*x +3),x)
print(a)

#내 Repo tensor101의 Dowehavenumpy를 확인할것.

#벡터가 입력인 다변수 함수의 경우 편미분을 사용한다.

#그래디언트 벡터란: 다차원 공간의 함수를 편미분하여 나타나는 극솟값으로 향하는 벡터...
#각 점에서 가장 빠르게 감소하는 방향과 같다.
#-델F는 델(-F)와 같다. 이는 각 점에서 가장 빠르게 감소하는 방향과 같다. 
"""Pseudo-code for GD
Input:gradient, init,lr,eps,Output:var
#gradient: 그래디언트 벡터 계싼하는 함수
#init: 시작점, lr: 학습률, eps: 알고리즘 종료 조건
var = init
grad = gradient(var)
while(norm(grad)>eps):
    var = var- lr*grad
    grad = gradient(var)
"""

