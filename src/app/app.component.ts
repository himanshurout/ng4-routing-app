import { Component, OnInit } from '@angular/core';
import { NgModel } from '@angular/forms';

@Component({
  selector: 'app-root',
  templateUrl: './app.component.html',
  styleUrls: ['./app.component.css']
})
export class AppComponent  implements OnInit{
  name:string;
  email:string;
  phone:string;
  user: User;
  users: User[];
  editIndex: number;
  buttonText: string;

  ngOnInit(){
    this.buttonText = "Add";
    this.user = new User("","","");
  	this.users = [];
  	let newUser = new User("Gitanjali","gitanjali@gmail.com","9439901000");
  	this.users.push(newUser);
  	newUser = new User("Manjulata","manjulata@gmail.com","9866590000");
  	this.users.push(newUser);
  }

  saveUser(){
    if(this.buttonText == 'Add'){
      this.users.push(this.user);
    }else{
      this.users[this.editIndex] = JSON.parse(JSON.stringify(this.user));
    }
    this.resetForm();
  }

  editUser(index, user){
    this.user = JSON.parse(JSON.stringify(user));
    this.editIndex = index;
    this.buttonText = "Update";
  }

  resetForm(){
    this.user = new User("","","");
    this.editIndex = null;
    this.buttonText = "Add";
  }

  deleteUser(index){
    this.users.splice(index, 1);
  }

}

class User {
	name:string;
  email:string;
	phone:string;

	constructor(name:string, email:string, phone:string){
		this.name = name;
    this.email = email;
		this.phone = phone;
	}
}