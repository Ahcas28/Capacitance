function res = int_trapez_3(integrand,NN,xmin,xmax,ymin,ymax,zmin,zmax)
    xs = linspace(xmin,xmax,NN);
    ys = linspace(ymin,ymax,NN);
    zs = linspace(zmin,zmax,NN);
    mesh = zeros(NN,NN,NN);
    for i = 1:NN
        for j =1: NN
            for k = 1:NN
            mesh(i,j,k) = integrand(xs(i),ys(j),zs(k));
            end
        end
    end
    res = trapz(zs,trapz(ys,trapz(xs,mesh,2)));
end