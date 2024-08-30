SELECT COUNT(*)
FROM comments AS c,
  posts AS p,
  postHistory AS ph,
  votes AS v,
  badges AS b,
  users AS u
WHERE u.Id = p.OwnerUserId
  AND u.Id = b.UserId
  AND p.Id = c.PostId
  AND p.Id = ph.PostId
  AND p.Id = v.PostId
  AND p.PostTypeId = 1
  AND p.Score <= 192
  AND p.ViewCount >= 0
  AND p.ViewCount <= 2772
  AND p.AnswerCount <= 5
  AND u.DownVotes >= 0;
