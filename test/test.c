int blub = 20;

int main() {
    int foo = 10;
    if (foo < 10) {
        foo -= 3;
    } else {
        foo = blub + foo;
    }
    return foo;
}