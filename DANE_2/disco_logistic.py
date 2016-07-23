def disco_logistic(X,Y,lambda,m,w_star,rounds,obj_star):
  #function [w,dist,distobj] =

  # Jealei Wang's code modified for python

    # digits(50);
    miter = 500
    [n,p] = np.shape(X)
    w = np.zeros([p,1])
    grad = np.zeros([p,1])
    bs = np.floor(n/m)
    iter = 0

    for k in xrange(m-1):
        wk, = logistic_newton( X[ (k-1)*bs+1:k*bs , : ], Y[ (k-1)*bs+1 : k*bs ] , lambda ) 
        w = w + (1/m)*wk

    wk = logistic_newton(X[ (m-1)*bs + 1 : end , : ] , Y[ (m-1)*bs + 1 : end ] , lambda )
    w = w + (1/m)*wk
    dist[iter] = np.linalg.norm( w - w_star  , order=1 )
    Qnew = sum( log( 1 + exp( np.dot(-Y , np.dot( X , w ) ) ) ) / n + 0.5 * lambda * np.linalg.norm( w , order = 1)^2

    distobj[ iter] = Qnew - obj_star


    for i in xrange(miter):
        eta = X*w;
        mu = 1./(1 + exp(-eta));
        grad = - (X'*(Y - mu))/n + lambda*w;
        H = bsxfun(@times,X,sqrt(mu.*(1-mu)));
        H = H'*H/n + lambda*eye(p);
        k = 1; 
        indexk = (k-1)*bs+1:k*bs;
        Xk = X(indexk,:);
        Hk = bsxfun(@times,Xk,sqrt(mu(indexk).*(1-mu(indexk))));
        Hk = Hk'*Hk/bs + lambda*eye(p);
        delta_w = zeros(p,1);
        A = inv(Hk)*H;
        b = Hk\grad;
#    A = H;
#    b = grad;
        rk = A*delta_w - b;
        pk = -rk;
        iter = iter + 1;
        for j = 1:10
            alphak = rk'*rk/(pk'*A*pk);
            delta_w = delta_w + alphak*pk;
            rkrk = rk'*rk;
            rk = rk + alphak*A*pk;
            betak = (rk'*rk)/(rkrk);
            pk = -rk + betak*pk;
            iter = iter + 1;
            fprintf('i:#d,j:#d,nt_error:#f\n',i,j,norm(H*delta_w-grad,2));
#           dist(iter) = norm(w - delta_w - w_star);
            Qnew = sum(log(1+exp(-(2*Y - 1).*(X*(w-delta_w)))))/n + 0.5*lambda*norm(w,2)^2;
#           distobj(iter) = Qnew - obj_star;
            if norm(H*delta_w-grad,2) < 0.1*norm(grad);
#               if j > 1 && norm(rk) < 1e-1;
                    fprintf('Inner Loops:#d\n',j);
                    break; 
                end
            if iter >= rounds;
                break;
            end
        end
#    delta_k = sqrt(delta_w'*H*delta_w);
#    for j = 1:10
#        dd_w = Hk\(grad - H*delta_w);
#        delta_w = delta_w + dd_w;
#        iter = iter + 1;
#        dist(iter) = norm(w - delta_w - w_star);
#        Qnew = sum(log(1+exp(-(2*Y - 1).*(X*(w-delta_w)))))/n + 0.5*lambda*norm(w,2)^2;
#        distobj(iter) = Qnew - obj_star;
#        if j > 1 && norm(dd_w) < 0.1*norm(delta_w)
#           fprintf('Inner Loops:#d\n',j);
#           break; 
#        end
#        if iter >= rounds;
#           break;
#        end
#    end
#    w = w - (1/(1+delta_k))*delta_w;
#    dist(iter) = norm(w - w_star);
        w = w - delta_w;
        dist(iter) = norm(w - w_star);
        if iter >= rounds;
            break;
        end
    end


end










function [w,dist,distobj] = disco_logistic(X,Y,lambda,m,w_star,rounds,obj_star)

% digits(50);
miter = 500;
[n,p] = size(X);
w = zeros(p,1);
grad = zeros(p,1);
bs = floor(n/m);
iter = 1;

for k = 1:m-1
    [wk,~] = logistic_newton(X((k-1)*bs+1:k*bs,:),Y((k-1)*bs+1:k*bs),lambda);
    w = w + (1/m)*wk;
end
wk = logistic_newton(X((m-1)*bs+1:end,:),Y((m-1)*bs+1:end),lambda);
w = w + (1/m)*wk;
dist(iter) = norm(w - w_star);
Qnew = sum(log(1+exp(-Y.*(X*w))))/n + 0.5*lambda*norm(w,2)^2;
distobj(iter) = Qnew - obj_star;

for i = 1:miter
   eta = X*w;
   mu = 1./(1 + exp(-eta));
   grad = - (X'*(Y - mu))/n + lambda*w;
   H = bsxfun(@times,X,sqrt(mu.*(1-mu)));
   H = H'*H/n + lambda*eye(p);
   k = 1;
   indexk = (k-1)*bs+1:k*bs;
   Xk = X(indexk,:);
   Hk = bsxfun(@times,Xk,sqrt(mu(indexk).*(1-mu(indexk))));
   Hk = Hk'*Hk/bs + lambda*eye(p);
   delta_w = zeros(p,1);
   A = inv(Hk)*H;
   b = Hk\grad;
%    A = H;
%    b = grad;
   rk = A*delta_w - b;
   pk = -rk;
   iter = iter + 1;
   for j = 1:10
       alphak = rk'*rk/(pk'*A*pk);
       delta_w = delta_w + alphak*pk;
       rkrk = rk'*rk;
       rk = rk + alphak*A*pk;
       betak = (rk'*rk)/(rkrk);
       pk = -rk + betak*pk;
       iter = iter + 1;
       fprintf('i:%d,j:%d,nt_error:%f\n',i,j,norm(H*delta_w-grad,2));
%        dist(iter) = norm(w - delta_w - w_star);
       Qnew = sum(log(1+exp(-(2*Y - 1).*(X*(w-delta_w)))))/n + 0.5*lambda*norm(w,2)^2;
%        distobj(iter) = Qnew - obj_star;
       if norm(H*delta_w-grad,2) < 0.1*norm(grad);
%        if j > 1 && norm(rk) < 1e-1;
          fprintf('Inner Loops:%d\n',j);
          break;
       end
       if iter >= rounds;
          break;
       end
   end
%    delta_k = sqrt(delta_w'*H*delta_w);
%    for j = 1:10
%        dd_w = Hk\(grad - H*delta_w);
%        delta_w = delta_w + dd_w;
%        iter = iter + 1;
%        dist(iter) = norm(w - delta_w - w_star);
%        Qnew = sum(log(1+exp(-(2*Y - 1).*(X*(w-delta_w)))))/n + 0.5*lambda*norm(w,2)^2;
%        distobj(iter) = Qnew - obj_star;
%        if j > 1 && norm(dd_w) < 0.1*norm(delta_w)
%           fprintf('Inner Loops:%d\n',j);
%           break;
%        end
%        if iter >= rounds;
%           break;
%        end
%    end
%    w = w - (1/(1+delta_k))*delta_w;
%    dist(iter) = norm(w - w_star);
   w = w - delta_w;
   dist(iter) = norm(w - w_star);
   if iter >= rounds;
      break;
   end
end


end