function res = int_trapez(integrand,NN,xmin,xmax,ymin,ymax)
    xs = linspace(xmin,xmax,NN);
    ys = linspace(ymin,ymax,NN);
    mesh = zeros(NN,NN);
    for i = 1:NN
        for j =1: NN
            mesh(i,j) = integrand(xs(i),ys(j));
        end
    end
    res = trapz(ys,trapz(xs,mesh,2));
end