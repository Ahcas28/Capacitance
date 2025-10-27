function produce_condensation(cx,modes_skin,N)
    cx_dup = cx;
    % degree of localization
    N_cx = [];
    N_sum = [];
    cx_proj = [];
    j = 1;
    while ~isempty(cx_dup)
        cx_proj = [cx_proj cx_dup(1)];
        N_cx_i = sum(cx_dup(:) == cx_dup(1));
        N_cx = [N_cx; N_cx_i];
        N_sum = [N_sum sum(N_cx)];
        cx_dup = cx_dup(N_cx_i+1:end);
        j = j+1;
    end
    
    L = size(N_cx,1);
    deg_loc = zeros(L,N);
    for j = 1:N
        mode_j = real(modes_skin(:,j));
        norm_j = norm(mode_j);
        for i = 1:L
            deg_loc(i,j) = norm(mode_j(1:N_sum(i)))/norm_j;
        end
    end
    
    figure
    hold on
    for j = 1:N
        plot(cx_proj,mean(deg_loc,2),'r','linewidth',3)
        plot(cx_proj,deg_loc(:,j),'color', [.5 .5 .5], 'linewidth', 1.5)
    end
    xlabel('Position of the resonators','interpreter','latex','FontSize',20)
    ylabel('Degree of condensation','interpreter','latex','FontSize',20)
    set(gca,'TickLabelInterpreter','latex','FontSize',20)
    set(gca,'XTick',[-25:10:25])
    xlim([min(cx),max(cx)])
end

