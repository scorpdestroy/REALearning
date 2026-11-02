import pytest
from main import System

def test_init():
    system = System()
    
def test_register():
    system = System()
    
    assert system.register("user1", "password1") == True
    
def test_login():
    system = System()
    system.register("user1", "password1")
    assert system.login("user1", "password1") == True
    
def test_login_2():
    """тест на логин незарегестрированного пользователя"""
    system = System()
    
    assert system.login("not_real", "pass") == False
    
def test_login_3():
    """неправильный пароль"""
    system = System()
    system.register("user1", "password1")
    assert system.login("user1", "pass") == False
    
def test_login_4():
    """второй пользователь"""
    system = System()
    system.register("user2", "password2")
    assert system.login("user2", "password2") == True
    
def test_login_5():
    """вход без регистрации"""
    system = System()
    assert system.login("user2", "password2") == False
    
def test_login_6():
    """регистрация и вход нескольких пользователей"""
    system = System()
    system.register("user1", "password1")
    system.register("user2", "password2")
    assert system.login("user1", "password1") == True
    
def test_login_7():
    """пароль другого пользователя не подходит"""
    system = System()
    system.register("user1", "password1")
    system.register("user2", "password2")

    assert system.login("user1", "password2") == False
    
def test_login_8():
    """регистрация кучи пользователей"""
    system = System()

    for i in range(100):
        system.register(f"user{i}", f"password{i}")

    for i in range(100):
        assert system.login(f"user{i}", f"password{i}") == True
        
def test_login_9():
    """повторная регистрация"""
    system = System()
    system.register("user1", "password1")

    assert system.register("user1", "password2") == False
    
def test_login_10():
    """просмотр профиля без входа"""
    system = System()
    system.register("user1", "password1")

    with pytest.raises(PermissionError):
        system.get_profile()
        
def test_login_11():
    """просмотр профиля после авторизации"""
    system = System()
    system.register("user1", "password1")
    system.login("user1", "password1")

    assert system.get_profile() == {"name": "user1"}
    
def test_login_12():
    """просмотр профиля другого пользователя"""
    system = System()
    system.register("user2", "password2")
    system.login("user2", "password2")

    assert system.get_profile() == {"name": "user2"}
    
def test_login_13():
    """выход из системы"""
    system = System()
    system.register("user1", "password1")
    system.login("user1", "password1")

    assert system.logout() is True
    
def test_login_14():
    """просмотр профиля после выхода"""
    system = System()
    system.register("user1", "password1")
    system.login("user1", "password1")
    system.logout()

    with pytest.raises(PermissionError):
        system.get_profile()
        
def test_login_15():
    """изменение пароля"""
    system = System()
    system.register("user1", "password1")
    system.login("user1", "password1")

    assert system.change_password("newpassword") == True