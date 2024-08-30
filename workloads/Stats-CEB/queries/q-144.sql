SELECT COUNT(*)
FROM comments AS c,
  posts AS p,
  postHistory AS ph,
  votes AS v,
  badges AS b,
  users AS u
WHERE u.Id = p.OwnerUserId
  AND p.Id = v.PostId
  AND p.Id = c.PostId
  AND u.Id = b.UserId
  AND p.Id = ph.PostId
  AND p.AnswerCount >= 0
  AND p.CommentCount >= 0
  AND b.Date <= CAST('2014-09-11 21:46:02' AS timestamp)
  AND u.Reputation >= 1
  AND u.Reputation <= 642
  AND u.DownVotes >= 0;
