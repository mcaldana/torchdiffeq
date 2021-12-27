import torch

class Print(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, op, final, unused):
        ctx.op = op
        ctx.final = final
        ctx.unused = unused
        return x

    @staticmethod
    def backward(ctx, x):
        print('op={}, final={}, unused={} has a bug'.format(ctx.op, ctx.final, ctx.unused))
        return x, None, None, None

def f(op, final, unused):
    x = torch.rand(1, requires_grad=True)
    if op == 'mul':
        x = 2 * x
    elif op == 'add':
        x = 2 + x
    elif op == 'relu':
        x = x.relu()
    elif op == 'clone':
        x = x.clone()
    elif op == 'view':
        x = x.view(-1)
    x = Print.apply(x, op, final, unused)

    y = torch.rand(1, requires_grad=True)

    if final:
        x = y + x

    try:
        torch.autograd.grad((x,), (y,), allow_unused=unused)
    except RuntimeError:
        pass


f('mul', False, True)
f('add', False, True)
f('relu', False, True)
f('clone', False, True)
f('view', False, True)

f('mul', True, True)
f('add', True, True)
f('relu', True, True)
f('clone', True, True)
f('view', True, True)

f('mul', True, False)
f('add', True, False)
f('relu', True, False)
f('clone', True, False)
f('view', True, False)

f('mul', False, False)
f('add', False, False)
f('relu', False, False)
f('clone', False, False)
f('view', False, False)