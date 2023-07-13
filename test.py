
import torch
conf_data =torch.tensor \
    ([[1, 2], [4, 5]])

conf_t =torch.tensor([[3, 4], [3, 4]])
# conf_data = conf_data.view(-1)

# conf_t = conf_t.view(-1)


print(conf_data)
print(conf_t)


ans = conf_data @ conf_t
ans1 = torch.mm(conf_data, conf_t)

print(ans)
print(ans1)

print(conf_data ** 2)
